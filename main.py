import argparse
import copy
import random

import numpy as np
import torch
import torch.nn as nn
from plyfile import PlyData
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(ply_path: str, max_points: int):
    plydata = PlyData.read(ply_path)
    data = plydata["vertex"].data

    xyz_raw = np.vstack([data["x"], data["y"], data["z"]]).T.astype(np.float32)

    # Keep all numeric point attributes except known target columns.
    feature_columns = [name for name in data.dtype.names if name not in {"class", "label"}]
    features_raw = np.vstack([data[name] for name in feature_columns]).T.astype(np.float32)
    labels_raw = data["class"].astype(int)

    print(f"Original points: {features_raw.shape[0]} | feature_dim: {features_raw.shape[1]}")
    print(f"Feature columns: {feature_columns}")

    n_points = features_raw.shape[0]
    sample_size = min(max_points, n_points)
    sampled_idx = np.random.choice(n_points, sample_size, replace=False) if sample_size < n_points else np.arange(n_points)

    xyz_raw = xyz_raw[sampled_idx]
    features_raw = features_raw[sampled_idx]
    labels_raw = labels_raw[sampled_idx]
    print(f"After sampling: ({features_raw.shape[0]}, {features_raw.shape[1]})")

    # Stratified split requires at least 2 samples per class in each split step.
    raw_unique, raw_counts = np.unique(labels_raw, return_counts=True)
    valid_raw_classes = raw_unique[raw_counts >= 3]
    rare_class_count = int(np.sum(raw_counts < 3))
    if rare_class_count > 0:
        keep_mask = np.isin(labels_raw, valid_raw_classes)
        removed_points = int((~keep_mask).sum())
        xyz_raw = xyz_raw[keep_mask]
        features_raw = features_raw[keep_mask]
        labels_raw = labels_raw[keep_mask]
        print(
            f"Removed {removed_points} points from {rare_class_count} ultra-rare classes "
            "(<3 samples) to enable stable stratified splits."
        )

    unique_labels = np.unique(labels_raw)
    label_map = {old: i for i, old in enumerate(unique_labels)}
    labels = np.array([label_map[x] for x in labels_raw], dtype=np.int64)
    print(f"Classes: {len(unique_labels)}")

    return xyz_raw, features_raw, labels, unique_labels


class PointMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def evaluate(model, loader, device, loss_fn):
    model.eval()
    all_preds = []
    all_true = []
    running_loss = 0.0
    total = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            running_loss += loss.item() * xb.size(0)
            total += xb.size(0)

            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_true.append(yb.cpu().numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_preds)
    return running_loss / max(1, total), accuracy_score(y_true, y_pred), y_true, y_pred


def main():
    parser = argparse.ArgumentParser(description="Accuracy-focused point cloud classification")
    parser.add_argument("--ply", type=str, default="data/Lille1.ply", help="Input .ply file")
    parser.add_argument("--max-points", type=int, default=120000, help="Max points to sample")
    parser.add_argument("--epochs", type=int, default=120, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=2048, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--baseline-accuracy",
        type=float,
        default=0.2989,
        help="Baseline test accuracy used to report optimization gain",
    )
    parser.add_argument(
        "--visualize",
        dest="visualize",
        action="store_true",
        help="Visualize predicted labels at the end (default: enabled)",
    )
    parser.add_argument(
        "--no-visualize",
        dest="visualize",
        action="store_false",
        help="Disable final visualization window",
    )
    parser.add_argument(
        "--balanced-loss",
        action="store_true",
        help="Use class-weighted loss (better minority recall, may reduce overall accuracy)",
    )
    parser.set_defaults(visualize=True)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    xyz_raw, features_raw, labels, unique_labels = load_data(args.ply, args.max_points)
    num_classes = len(unique_labels)
    input_dim = features_raw.shape[1]

    all_idx = np.arange(len(features_raw))
    train_idx, test_idx = train_test_split(
        all_idx, test_size=0.2, random_state=args.seed, stratify=labels
    )
    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=0.125,
        random_state=args.seed,
        stratify=labels[train_idx],
    )

    train_points = features_raw[train_idx]
    val_points = features_raw[val_idx]
    test_points = features_raw[test_idx]

    train_labels = labels[train_idx]
    val_labels = labels[val_idx]
    test_labels = labels[test_idx]

    mean = train_points.mean(axis=0, keepdims=True)
    std = train_points.std(axis=0, keepdims=True) + 1e-6

    train_points = (train_points - mean) / std
    val_points = (val_points - mean) / std
    test_points = (test_points - mean) / std

    X_train = torch.tensor(train_points, dtype=torch.float32)
    y_train = torch.tensor(train_labels, dtype=torch.long)
    X_val = torch.tensor(val_points, dtype=torch.float32)
    y_val = torch.tensor(val_labels, dtype=torch.long)
    X_test = torch.tensor(test_points, dtype=torch.float32)
    y_test = torch.tensor(test_labels, dtype=torch.long)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_val, y_val),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_test, y_test),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    if args.balanced_loss:
        class_counts = np.bincount(train_labels, minlength=num_classes)
        class_weights = len(train_labels) / (num_classes * np.maximum(class_counts, 1))
        class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        print("Using class-weighted loss.")
    else:
        loss_fn = nn.CrossEntropyLoss()
        print("Using standard cross-entropy loss.")

    model = PointMLP(input_dim=input_dim, num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    best_state = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    print("\nTraining...\n")

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        seen = 0
        train_true = []
        train_pred = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            seen += xb.size(0)
            pred = logits.argmax(dim=1)
            train_true.append(yb.detach().cpu().numpy())
            train_pred.append(pred.detach().cpu().numpy())

        train_acc = accuracy_score(np.concatenate(train_true), np.concatenate(train_pred))
        train_loss = running_loss / max(1, seen)

        val_loss, val_acc, _, _ = evaluate(model, val_loader, device, loss_fn)
        scheduler.step(val_loss)

        improved = val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss)
        if improved:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f} | lr={current_lr:.6f}"
        )

        if epochs_without_improvement >= args.patience:
            print(f"\nEarly stopping triggered after epoch {epoch + 1}.")
            break

    model.load_state_dict(best_state)

    print("\nEvaluating best model on test split...\n")
    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, device, loss_fn)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    if args.visualize:
        try:
            import open3d as o3d
        except ImportError:
            print("Open3D is not installed. Skipping visualization.")
            return

        all_points_norm = (features_raw - mean) / std
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(all_points_norm, dtype=torch.float32, device=device))
            all_preds = logits.argmax(dim=1).cpu().numpy()

        rng = np.random.default_rng(0)
        colors = rng.random((num_classes, 3), dtype=np.float32)
        point_colors = colors[all_preds]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz_raw)
        pcd.colors = o3d.utility.Vector3dVector(point_colors)

        print("\nRendering predicted segmentation...")
        o3d.visualization.draw_geometries([pcd])

    baseline_acc = args.baseline_accuracy
    abs_gain = test_acc - baseline_acc
    rel_gain = (abs_gain / baseline_acc * 100.0) if baseline_acc > 0 else float("inf")

    print("\nOptimization Summary:")
    print(f"Baseline Accuracy: {baseline_acc:.4f}")
    print(f"Current Accuracy:  {test_acc:.4f}")
    print(f"Absolute Gain:     {abs_gain:+.4f} ({abs_gain * 100:+.2f} percentage points)")
    if np.isfinite(rel_gain):
        print(f"Relative Gain:     {rel_gain:+.2f}%")
    else:
        print("Relative Gain:     undefined (baseline accuracy is 0)")


if __name__ == "__main__":
    main()
