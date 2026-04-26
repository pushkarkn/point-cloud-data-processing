# Data Point Cloud Optimization

Point-cloud classification pipeline using PyTorch and PLY input data.

This project trains a neural network on point attributes from a `.ply` file, evaluates performance, reports optimization gains over a baseline, and visualizes predicted segmentation in Open3D.

## 1. Project Goal

- Train a classifier for point-cloud semantic classes.
- Improve test accuracy through better feature usage and training strategy.
- Show final point-cloud prediction visually.
- Report how much optimization was achieved compared to a baseline accuracy.

## 2. Project Structure

- `main.py`: Full training, evaluation, optimization summary, and visualization pipeline.
- `data/`: Input `.ply` dataset files.
- `output.txt`: Optional captured output/log file.

## 3. Data and Features

Input file default:

- `data/Lille1.ply`

Target column:

- `class`

Features used for training:

- `x`, `y`, `z`, `x_origin`, `y_origin`, `z_origin`, `GPS_time`, `reflectance`

Notes:

- Labels are remapped to contiguous class IDs `[0..num_classes-1]`.
- Very rare classes (fewer than 3 sampled points) are removed to keep stratified splitting stable.

## 4. Training Flow

1. Load `.ply` point cloud.
2. Sample up to `--max-points` points.
3. Build feature matrix and labels.
4. Remove ultra-rare classes for robust split.
5. Split data into train/validation/test with stratification.
6. Normalize features using train-set mean/std.
7. Train an MLP classifier with:
   - BatchNorm
   - Dropout
   - AdamW optimizer
   - ReduceLROnPlateau scheduler
   - Early stopping by validation performance
8. Evaluate on test set.
9. Print metrics:
   - Test loss and test accuracy
   - Classification report
   - Confusion matrix
10. Render predicted segmentation in Open3D (enabled by default).
11. Print optimization summary against baseline accuracy.

## 5. Environment Setup

From project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install numpy torch plyfile scikit-learn open3d
```

## 6. Run Commands

Default run (training + evaluation + visualization + optimization summary):

```powershell
& "c:/Coding/Data Point Cloud Optimization/.venv/Scripts/python.exe" "c:/Coding/Data Point Cloud Optimization/main.py" --epochs 35 --patience 8
```

Run without visualization window:

```powershell
& "c:/Coding/Data Point Cloud Optimization/.venv/Scripts/python.exe" "c:/Coding/Data Point Cloud Optimization/main.py" --epochs 35 --patience 8 --no-visualize
```

Use class-weighted loss (can help minority recall, may reduce overall accuracy):

```powershell
& "c:/Coding/Data Point Cloud Optimization/.venv/Scripts/python.exe" "c:/Coding/Data Point Cloud Optimization/main.py" --epochs 35 --patience 8 --balanced-loss
```

Set custom baseline for optimization summary:

```powershell
& "c:/Coding/Data Point Cloud Optimization/.venv/Scripts/python.exe" "c:/Coding/Data Point Cloud Optimization/main.py" --baseline-accuracy 0.2989
```

## 7. Optimization Summary Output

At the end of each run, the script prints:

- Baseline Accuracy
- Current Accuracy
- Absolute Gain
- Relative Gain (%)

Example interpretation:

- Baseline: 0.2989
- Current: 0.8535
- Absolute gain: +0.5546 (55.46 percentage points)
- Relative gain: +185.54%

## 8. Suggested Git Workflow

```powershell
git add main.py README.md
git commit -m "Add optimization summary and complete project README"
git push origin <your-branch>
```

## 9. Troubleshooting

- If Open3D import warning appears in editor but runtime works, reload VS Code/Python interpreter.
- If stratified split errors appear, increase `--max-points` or keep rare-class filtering enabled.
- If training is slow on CPU, reduce `--max-points` or `--epochs`, or run on GPU.
