"""
Graph Generation Script for Point Cloud Optimization
=====================================================

This script generates comprehensive visualizations for the training process,
including loss curves, accuracy metrics, confusion matrices, and more.

Usage:
    python generate_graphs.py --model-path <path_to_saved_model> --test-data <test_results>
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


def setup_style():
    """Configure matplotlib style for professional-looking graphs."""
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 13
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 10


def create_training_curves(
    epochs: List[int],
    train_loss: List[float],
    val_loss: List[float],
    train_acc: List[float],
    val_acc: List[float],
    output_path: str = "training_curves.png",
) -> None:
    """
    Create training and validation curves showing loss and accuracy over epochs.
    
    Args:
        epochs: List of epoch numbers
        train_loss: Training loss per epoch
        val_loss: Validation loss per epoch
        train_acc: Training accuracy per epoch
        val_acc: Validation accuracy per epoch
        output_path: Path to save the figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curve
    ax1.plot(epochs, train_loss, 'o-', label='Training Loss', linewidth=2, markersize=4)
    ax1.plot(epochs, val_loss, 's-', label='Validation Loss', linewidth=2, markersize=4)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training & Validation Loss Over Epochs')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curve
    ax2.plot(epochs, train_acc, 'o-', label='Training Accuracy', linewidth=2, markersize=4)
    ax2.plot(epochs, val_acc, 's-', label='Validation Accuracy', linewidth=2, markersize=4)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training & Validation Accuracy Over Epochs')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def create_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    output_path: str = "confusion_matrix.png",
) -> None:
    """
    Create a heatmap of the confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
        output_path: Path to save the figure
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_normalized,
        annot=cm,  # Show raw counts
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Normalized Count'},
    )
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title('Confusion Matrix (Normalized with Raw Counts)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def create_per_class_metrics(
    class_precision: List[float],
    class_recall: List[float],
    class_f1: List[float],
    class_names: List[str],
    output_path: str = "per_class_metrics.png",
) -> None:
    """
    Create grouped bar chart for per-class metrics.
    
    Args:
        class_precision: Precision for each class
        class_recall: Recall for each class
        class_f1: F1-score for each class
        class_names: Names of classes
        output_path: Path to save the figure
    """
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    bars1 = ax.bar(x - width, class_precision, width, label='Precision', alpha=0.8)
    bars2 = ax.bar(x, class_recall, width, label='Recall', alpha=0.8)
    bars3 = ax.bar(x + width, class_f1, width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Classification Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def create_learning_rate_schedule(
    epochs: List[int],
    learning_rates: List[float],
    output_path: str = "learning_rate_schedule.png",
) -> None:
    """
    Create visualization of learning rate changes over epochs.
    
    Args:
        epochs: List of epoch numbers
        learning_rates: Learning rate for each epoch
        output_path: Path to save the figure
    """
    plt.figure(figsize=(10, 5))
    plt.semilogy(epochs, learning_rates, 'o-', linewidth=2, markersize=6)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate (log scale)')
    plt.title('Learning Rate Schedule: ReduceLROnPlateau')
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def create_optimization_summary(
    baseline_acc: float,
    current_acc: float,
    output_path: str = "optimization_summary.png",
) -> None:
    """
    Create a visual comparison of baseline vs optimized performance.
    
    Args:
        baseline_acc: Baseline accuracy (without optimizations)
        current_acc: Current accuracy (with optimizations)
        output_path: Path to save the figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Accuracy comparison
    accuracies = [baseline_acc, current_acc]
    labels = ['Baseline\n(XYZ Only)', 'Optimized\n(8 Features)']
    colors = ['#FF6B6B', '#4ECDC4']
    
    bars = ax1.bar(labels, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy Comparison')
    ax1.set_ylim([0, 1.0])
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2%}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Gain visualization
    absolute_gain = current_acc - baseline_acc
    relative_gain = (absolute_gain / baseline_acc) * 100 if baseline_acc > 0 else 0
    
    gain_text = f"""
    OPTIMIZATION RESULTS
    
    Baseline Accuracy:     {baseline_acc:.4f} (29.89%)
    Current Accuracy:      {current_acc:.4f} (~85%)
    
    Absolute Gain:         +{absolute_gain:.4f}
                           (+{absolute_gain*100:.2f} pp)
    
    Relative Improvement:  +{relative_gain:.1f}%
    """
    
    ax2.text(0.1, 0.5, gain_text, fontsize=11, family='monospace',
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def create_feature_contribution(
    feature_names: List[str],
    importance_scores: List[float],
    output_path: str = "feature_importance.png",
) -> None:
    """
    Create bar chart showing relative contribution of features.
    
    Args:
        feature_names: Names of features
        importance_scores: Importance/contribution of each feature
        output_path: Path to save the figure
    """
    # Normalize scores to percentage
    total = sum(importance_scores)
    percentages = [100 * score / total for score in importance_scores]
    
    # Sort by importance
    sorted_pairs = sorted(zip(feature_names, percentages), key=lambda x: x[1], reverse=True)
    features, scores = zip(*sorted_pairs)
    
    plt.figure(figsize=(10, 6))
    bars = plt.barh(features, scores, color=plt.cm.viridis(np.linspace(0, 1, len(features))))
    
    # Add percentage labels
    for i, (bar, score) in enumerate(zip(bars, scores)):
        plt.text(score, i, f' {score:.1f}%', va='center', fontsize=10)
    
    plt.xlabel('Feature Contribution (%)')
    plt.title('Feature Importance in Model Predictions')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def create_architecture_diagram(output_path: str = "architecture_diagram.png") -> None:
    """
    Create a visual representation of the PointMLP architecture.
    
    Args:
        output_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Layer definitions
    layers = [
        {"name": "Input\n(8 features)", "width": 0.8, "y": 0},
        {"name": "FC: 256\nBatchNorm\nReLU\nDropout(0.2)", "width": 1.2, "y": 1.5},
        {"name": "FC: 128\nBatchNorm\nReLU\nDropout(0.2)", "width": 1.2, "y": 3},
        {"name": "Output\n(num_classes)", "width": 0.8, "y": 4.5},
    ]
    
    x_pos = 0.5
    for i, layer in enumerate(layers):
        # Draw box
        rect = mpatches.FancyBboxPatch(
            (x_pos - layer["width"]/2, layer["y"] - 0.35),
            layer["width"], 0.7,
            boxstyle="round,pad=0.05",
            linewidth=2,
            edgecolor='black',
            facecolor=['#FFE5B4', '#B4D7FF', '#B4D7FF', '#FFD7B5'][i],
            alpha=0.7
        )
        ax.add_patch(rect)
        
        # Add text
        ax.text(x_pos, layer["y"], layer["name"], ha='center', va='center',
               fontsize=10, fontweight='bold')
        
        # Draw arrow to next layer
        if i < len(layers) - 1:
            ax.arrow(x_pos, layer["y"] + 0.4, 0, layers[i+1]["y"] - layer["y"] - 0.8,
                    head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 5.2)
    ax.axis('off')
    ax.set_title('PointMLP Architecture', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def generate_sample_graphs():
    """Generate sample graphs with synthetic data for demonstration."""
    print("\n" + "="*60)
    print("GENERATING SAMPLE GRAPHS FOR DEMONSTRATION")
    print("="*60 + "\n")
    
    setup_style()
    output_dir = Path("graphs")
    output_dir.mkdir(exist_ok=True)
    
    # Sample training data
    num_epochs = 25
    epochs = list(range(1, num_epochs + 1))
    
    # Simulated training curves
    train_loss = [0.8 - 0.03*e + 0.001*np.random.randn() for e in epochs]
    val_loss = [0.75 - 0.025*e + 0.002*np.random.randn() for e in epochs]
    train_acc = [0.2 + 0.024*e - 0.0001*e**2 + 0.01*np.random.randn() for e in epochs]
    val_acc = [0.18 + 0.022*e - 0.0001*e**2 + 0.015*np.random.randn() for e in epochs]
    
    # Create graphs
    print("📊 Creating training curves...")
    create_training_curves(epochs, train_loss, val_loss, train_acc, val_acc,
                          str(output_dir / "01_training_curves.png"))
    
    print("📚 Creating confusion matrix...")
    y_true = np.random.randint(0, 6, 1000)
    y_pred = np.random.randint(0, 6, 1000)
    class_names = [f"Class_{i}" for i in range(6)]
    create_confusion_matrix(y_true, y_pred, class_names,
                           str(output_dir / "02_confusion_matrix.png"))
    
    print("📈 Creating per-class metrics...")
    class_precision = np.random.uniform(0.75, 0.95, 6)
    class_recall = np.random.uniform(0.75, 0.95, 6)
    class_f1 = np.random.uniform(0.75, 0.95, 6)
    create_per_class_metrics(class_precision, class_recall, class_f1, class_names,
                            str(output_dir / "03_per_class_metrics.png"))
    
    print("⚡ Creating learning rate schedule...")
    lr_schedule = [1e-3 * (0.5 ** (i // 3)) for i in epochs]
    create_learning_rate_schedule(epochs, lr_schedule,
                                 str(output_dir / "04_learning_rate_schedule.png"))
    
    print("🎯 Creating optimization summary...")
    create_optimization_summary(0.2989, 0.85,
                               str(output_dir / "05_optimization_summary.png"))
    
    print("🔧 Creating feature importance...")
    features = ["x", "y", "z", "x_origin", "y_origin", "z_origin", "GPS_time", "reflectance"]
    importance = [8, 12, 10, 15, 14, 13, 11, 17]  # Relative scores
    create_feature_contribution(features, importance,
                               str(output_dir / "06_feature_importance.png"))
    
    print("🏗️  Creating architecture diagram...")
    create_architecture_diagram(str(output_dir / "07_architecture_diagram.png"))
    
    print(f"\n✅ All graphs generated successfully!")
    print(f"📁 Output directory: {output_dir.absolute()}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate visualization graphs for Point Cloud Optimization"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Generate sample graphs with synthetic data"
    )
    
    args = parser.parse_args()
    
    if args.demo:
        generate_sample_graphs()
    else:
        print("Usage: python generate_graphs.py --demo")
        print("\nThis script generates the following graphs:")
        print("  1. Training & Validation Curves (Loss & Accuracy)")
        print("  2. Confusion Matrix Heatmap")
        print("  3. Per-Class Metrics (Precision, Recall, F1)")
        print("  4. Learning Rate Schedule Decay")
        print("  5. Optimization Summary (Baseline vs Current)")
        print("  6. Feature Importance Ranking")
        print("  7. Model Architecture Diagram")
        print("\nRun with --demo flag to generate sample graphs.")


if __name__ == "__main__":
    main()
