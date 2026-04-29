import argparse
import random
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from plyfile import PlyData
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

try:
    import open3d as o3d
except Exception:
    o3d = None


DEFAULT_FEATURES = [
    "x",
    "y",
    "z",
    "x_origin",
    "y_origin",
    "z_origin",
    "GPS_time", 
    "reflectance",
]


class PointMLP(nn.Module):
    def __init__(self, in_features: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Point-cloud classification and optimization summary")
    parser.add_argument("--data", type=str, default="data/Lille1.ply", help="Path to .ply file")
    parser.add_argument("--target", type=str, default="class", help="Target label field in PLY")
    parser.add_argument("--max-points", type=int, default=200000, help="Max points to sample")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--val-size", type=float, default=0.2, help="Validation split ratio from train pool")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size")
    parser.add_argument("--epochs", type=int, default=35, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--balanced-loss", action="store_true", help="Use class-weighted cross entropy")
    parser.add_argument("--no-visualize", action="store_true", help="Disable Open3D visualization")
    parser.add_argument("--baseline-accuracy", type=float, default=0.2989, help="Baseline for optimization summary")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_ply(path: str) -> PlyData:
    return PlyData.read(path)


def extract_features_and_labels(
    ply: PlyData,
    feature_names: List[str],
    target_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    vertex = ply["vertex"].data
    available = set(vertex.dtype.names)

    missing = [col for col in feature_names if col not in available]
    if missing:
        raise ValueError(f"Missing feature columns in PLY: {missing}")
    if target_name not in available:
        raise ValueError(f"Target column '{target_name}' not found in PLY")

    features = np.column_stack([vertex[name].astype(np.float32) for name in feature_names]).astype(np.float32)
    labels_raw = np.asarray(vertex[target_name])
    return features, labels_raw


def sample_points(x: np.ndarray, y: np.ndarray, max_points: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(y)
    if max_points <= 0 or n <= max_points:
        idx = np.arange(n)
        return x, y, idx

    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_points, replace=False)
    return x[idx], y[idx], idx


def remove_ultra_rare_classes(
    x: np.ndarray,
    y: np.ndarray,
    keep_index: np.ndarray,
    min_count: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = Counter(y.tolist())
    keep_classes = {cls for cls, count in counts.items() if count >= min_count}
    keep_mask = np.array([label in keep_classes for label in y], dtype=bool)
    return x[keep_mask], y[keep_mask], keep_index[keep_mask]


def remap_labels(y: np.ndarray) -> Tuple[np.ndarray, Dict[int, int], Dict[int, int]]:
    unique = np.unique(y)
    old_to_new = {int(old): int(new) for new, old in enumerate(unique)}
    new_to_old = {v: k for k, v in old_to_new.items()}
    y_remapped = np.array([old_to_new[int(label)] for label in y], dtype=np.int64)
    return y_remapped, old_to_new, new_to_old


def normalize_with_train_stats(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (x_train - mean) / std, (x_val - mean) / std, (x_test - mean) / std, mean, std


def to_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        total_loss += loss.item() * yb.size(0)
        total_correct += (preds == yb).sum().item()
        total += yb.size(0)

    return total_loss / total, total_correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    y_true_all = []
    y_pred_all = []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        preds = logits.argmax(dim=1)

        total_loss += loss.item() * yb.size(0)
        total_correct += (preds == yb).sum().item()
        total += yb.size(0)

        y_true_all.append(yb.cpu().numpy())
        y_pred_all.append(preds.cpu().numpy())

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    return total_loss / total, total_correct / total, y_true, y_pred


@torch.no_grad()
def predict_all(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int = 8192) -> np.ndarray:
    model.eval()
    preds_all = []
    x_tensor = torch.from_numpy(x).float()
    for start in range(0, x_tensor.shape[0], batch_size):
        xb = x_tensor[start : start + batch_size].to(device)
        logits = model(xb)
        preds_all.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds_all)


def visualize_predictions(ply: PlyData, sampled_idx: np.ndarray, preds: np.ndarray) -> None:
    if o3d is None:
        print("[Visualization] Open3D not available, skipping visualization.")
        return

    vertices = ply["vertex"].data
    xyz = np.column_stack([
        vertices["x"].astype(np.float32),
        vertices["y"].astype(np.float32),
        vertices["z"].astype(np.float32),
    ])
    xyz = xyz[sampled_idx]

    num_classes = int(preds.max()) + 1
    rng = np.random.default_rng(123)
    palette = rng.uniform(0.1, 0.95, size=(num_classes, 3)).astype(np.float32)
    colors = palette[preds]

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    print("[Visualization] Opening Open3D window...")
    o3d.visualization.draw_geometries([cloud], window_name="Predicted Segmentation")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    print("Loading PLY:", args.data)
    ply = load_ply(args.data)

    x_all, y_raw = extract_features_and_labels(ply, DEFAULT_FEATURES, args.target)
    x_sampled, y_sampled_raw, sampled_idx = sample_points(x_all, y_raw, args.max_points, args.seed)

    x_filtered, y_filtered_raw, sampled_idx = remove_ultra_rare_classes(
        x_sampled,
        y_sampled_raw,
        sampled_idx,
        min_count=3,
    )
    if len(y_filtered_raw) == 0:
        raise RuntimeError("No samples left after rare-class filtering.")

    y, _, new_to_old = remap_labels(y_filtered_raw)
    num_classes = int(np.unique(y).size)

    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x_filtered,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=y_train_val,
    )

    x_train, x_val, x_test, mean, std = normalize_with_train_stats(x_train, x_val, x_test)
    x_filtered_norm = (x_filtered - mean) / std

    train_loader = to_loader(x_train, y_train, args.batch_size, shuffle=True)
    val_loader = to_loader(x_val, y_val, args.batch_size, shuffle=False)
    test_loader = to_loader(x_test, y_test, args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PointMLP(in_features=x_train.shape[1], num_classes=num_classes, dropout=args.dropout).to(device)

    class_weights = None
    if args.balanced_loss:
        classes = np.unique(y_train)
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    best_val_acc = -1.0
    best_state = None
    wait = 0

    print(f"Training on {device} | classes={num_classes} | samples={len(y)}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_acc)

        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if wait >= args.patience:
            print(f"Early stopping at epoch {epoch} (patience={args.patience}).")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc, y_true_test, y_pred_test = evaluate(model, test_loader, criterion, device)
    print("\n=== Test Metrics ===")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    target_names = [f"class_{new_to_old[i]}" for i in range(num_classes)]
    print("\n=== Classification Report ===")
    print(classification_report(y_true_test, y_pred_test, target_names=target_names, digits=4, zero_division=0))

    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_true_test, y_pred_test))

    if not args.no_visualize:
        y_pred_all = predict_all(model, x_filtered_norm.astype(np.float32), device)
        visualize_predictions(ply, sampled_idx, y_pred_all)

    baseline = args.baseline_accuracy
    absolute_gain = test_acc - baseline
    relative_gain = (absolute_gain / baseline) * 100.0 if baseline > 0 else float("inf")

    print("\n=== Optimization Summary ===")
    print(f"Baseline Accuracy : {baseline:.4f}")
    print(f"Current Accuracy  : {test_acc:.4f}")
    print(f"Absolute Gain     : {absolute_gain:+.4f} ({absolute_gain * 100:+.2f} percentage points)")
    if np.isfinite(relative_gain):
        print(f"Relative Gain     : {relative_gain:+.2f}%")
    else:
        print("Relative Gain     : inf (baseline is 0)")


if __name__ == "__main__":
    main()