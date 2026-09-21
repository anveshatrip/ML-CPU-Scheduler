"""
xgboost_model.py — XGBoost Burst Time Predictor
==================================================
WHAT THIS DOES:
    Trains an XGBoost regression model to predict CPU burst times
    from process features (type, priority, I/O frequency, memory, threads).

WHY XGBOOST:
    XGBoost (eXtreme Gradient Boosting) is the gold standard for tabular
    (structured) data prediction. It works by building an ensemble of
    decision trees, where each new tree corrects the errors of the previous
    ones. It's fast, accurate, and handles mixed feature types well.

    Think of it like this: one decision tree might learn "if process_type is
    cpu_bound, burst is probably > 20". Another tree might refine this to
    "if cpu_bound AND high memory, burst is probably > 35". XGBoost combines
    hundreds of these trees for a precise prediction.

HOW IT FITS IN THE PROJECT:
    1. generator.py creates processes with KNOWN burst times
    2. This model trains on those (features → burst_time)
    3. At scheduling time, the ML-SJF scheduler calls predict()
       with a new process's features to get a PREDICTED burst
    4. The scheduler picks the process with the shortest PREDICTED burst

USAGE:
    # Train the model
    python -m module2_ml.xgboost_model train

    # Predict burst for a single process
    python -m module2_ml.xgboost_model predict
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'saved_models')
MODEL_PATH = os.path.join(MODEL_DIR, 'xgboost_model.pkl')
STATS_PATH = os.path.join(MODEL_DIR, 'xgboost_stats.json')

# Features used for prediction (these come from the Process dataclass)
# process_type gets one-hot encoded into 4 binary columns
FEATURE_COLUMNS = ['priority', 'io_frequency', 'memory_mb', 'num_threads']
PROCESS_TYPES = ['cpu_bound', 'io_bound', 'mixed', 'interactive']

# XGBoost hyperparameters
# These are tuned for our synthetic data — you can experiment with them
XGBOOST_PARAMS = {
    'n_estimators': 200,        # Number of trees in the ensemble
    'max_depth': 6,             # Max depth of each tree (prevents overfitting)
    'learning_rate': 0.1,       # Step size for gradient descent
    'subsample': 0.8,           # Use 80% of data for each tree (prevents overfitting)
    'colsample_bytree': 0.8,    # Use 80% of features for each tree
    'reg_alpha': 0.1,           # L1 regularization (sparsity)
    'reg_lambda': 1.0,          # L2 regularization (smoothness)
    'random_state': 42,         # Reproducibility
}


# ═══════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════
def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw process data into ML-ready features.

    WHAT THIS DOES:
    - One-hot encodes 'process_type' into 4 binary columns:
      process_type_cpu_bound, process_type_io_bound,
      process_type_mixed, process_type_interactive
    - Keeps numerical features as-is: priority, io_frequency, memory_mb, num_threads

    WHY ONE-HOT ENCODING:
    ML models need numbers, not strings. One-hot encoding converts a categorical
    variable like process_type="cpu_bound" into 4 binary columns:
        process_type_cpu_bound = 1
        process_type_io_bound  = 0
        process_type_mixed     = 0
        process_type_interactive = 0

    Args:
        df: DataFrame with columns from Process.to_dict()

    Returns:
        DataFrame with only the feature columns (ready for model input)
    """
    # One-hot encode process_type
    for ptype in PROCESS_TYPES:
        df[f'process_type_{ptype}'] = (df['process_type'] == ptype).astype(int)

    # Select final feature columns
    feature_cols = FEATURE_COLUMNS + [f'process_type_{t}' for t in PROCESS_TYPES]
    return df[feature_cols]


def prepare_features_single(process_dict: dict) -> np.ndarray:
    """
    Prepare features for a SINGLE process (used at prediction time).

    Args:
        process_dict: Dictionary with process attributes
                     (pid, process_type, priority, io_frequency, memory_mb, num_threads)

    Returns:
        1D numpy array of features, shape (8,)
    """
    features = [
        process_dict['priority'],
        process_dict['io_frequency'],
        process_dict['memory_mb'],
        process_dict['num_threads'],
    ]

    # One-hot encode process_type
    for ptype in PROCESS_TYPES:
        features.append(1 if process_dict['process_type'] == ptype else 0)

    return np.array(features).reshape(1, -1)


# ═══════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════
def train(data_path: str = 'data/processes.csv', test_size: float = 0.2):
    """
    Train the XGBoost model on process data.

    STEP BY STEP:
    1. Load process data from CSV
    2. Split into features (X) and target (y = cpu_burst)
    3. Split into train (80%) and test (20%) sets
    4. Train XGBoost regressor
    5. Evaluate on test set (MAE, RMSE, R²)
    6. Save model and stats to disk

    Args:
        data_path: Path to the processes CSV file
        test_size: Fraction of data to use for testing (0.2 = 20%)

    Returns:
        Dictionary with model performance metrics
    """
    print("=" * 60)
    print("  XGBoost Burst Time Predictor — Training")
    print("=" * 60)

    # Step 1: Load data
    print(f"\n[1/5] Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"      Loaded {len(df)} processes")

    # Step 2: Prepare features and target
    print("[2/5] Preparing features...")
    X = prepare_features(df.copy())
    y = df['cpu_burst']

    feature_names = X.columns.tolist()
    print(f"      Features: {feature_names}")
    print(f"      Target: cpu_burst (mean={y.mean():.1f}, std={y.std():.1f})")

    # Step 3: Train/test split
    print(f"[3/5] Splitting data ({int((1-test_size)*100)}% train / {int(test_size*100)}% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    print(f"      Train: {len(X_train)} samples, Test: {len(X_test)} samples")

    # Step 4: Train model
    print("[4/5] Training XGBoost model...")
    model = XGBRegressor(**XGBOOST_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False  # Set to True to see training progress
    )
    print("      Training complete!")

    # Step 5: Evaluate
    print("[5/5] Evaluating on test set...")
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n      ┌────────────────────────────┐")
    print(f"      │  MAE  (Mean Abs Error): {mae:>6.2f} │")
    print(f"      │  RMSE (Root MSE):       {rmse:>6.2f} │")
    print(f"      │  R² Score:              {r2:>6.3f} │")
    print(f"      └────────────────────────────┘")

    # Feature importances
    importances = dict(zip(feature_names, model.feature_importances_.tolist()))
    importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

    print("\n      Feature Importances:")
    for feat, imp in importances.items():
        bar = "█" * int(imp * 40)
        print(f"        {feat:<30} {imp:.3f} {bar}")

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\n[✓] Model saved to {MODEL_PATH}")

    # Save stats
    stats = {
        'mae': round(mae, 4),
        'rmse': round(rmse, 4),
        'r2': round(r2, 4),
        'feature_importances': {k: round(v, 4) for k, v in importances.items()},
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'params': XGBOOST_PARAMS,
    }
    with open(STATS_PATH, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"[✓] Stats saved to {STATS_PATH}")

    return stats


# ═══════════════════════════════════════════════════════════════════════
# PREDICTION
# ═══════════════════════════════════════════════════════════════════════
class XGBoostPredictor:
    """
    Wrapper for making burst time predictions with the trained XGBoost model.

    USAGE:
        predictor = XGBoostPredictor()
        predicted_burst = predictor.predict({
            'process_type': 'cpu_bound',
            'priority': 15,
            'io_frequency': 0.1,
            'memory_mb': 1024,
            'num_threads': 8,
        })
        print(f"Predicted burst: {predicted_burst}")
    """

    def __init__(self, model_path: str = MODEL_PATH):
        """Load the trained model from disk."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                f"Run 'python -m module2_ml.xgboost_model train' first."
            )
        self.model = joblib.load(model_path)

    def predict(self, process_dict: dict) -> float:
        """
        Predict burst time for a single process.

        Args:
            process_dict: Dictionary with process attributes

        Returns:
            Predicted CPU burst time (float, always >= 1)
        """
        features = prepare_features_single(process_dict)
        prediction = self.model.predict(features)[0]
        return max(1.0, float(prediction))  # Burst time can't be < 1

    def predict_batch(self, processes_df: pd.DataFrame) -> np.ndarray:
        """
        Predict burst times for a batch of processes.

        Args:
            processes_df: DataFrame with process data

        Returns:
            Array of predicted burst times
        """
        X = prepare_features(processes_df.copy())
        predictions = self.model.predict(X)
        return np.maximum(1.0, predictions)


# ═══════════════════════════════════════════════════════════════════════
# MODEL STATS (for dashboard)
# ═══════════════════════════════════════════════════════════════════════
def get_model_stats() -> dict:
    """
    Load and return saved model stats.
    Called by Module 3's /api/model-stats endpoint.
    """
    if os.path.exists(STATS_PATH):
        with open(STATS_PATH, 'r') as f:
            return json.load(f)
    return {'error': 'Model not trained yet. Run training first.'}


# ═══════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='XGBoost Burst Time Predictor')
    parser.add_argument('action', choices=['train', 'predict', 'stats'],
                        help='Action to perform')
    parser.add_argument('--data', type=str, default='data/processes.csv',
                        help='Path to training data CSV')
    args = parser.parse_args()

    if args.action == 'train':
        train(data_path=args.data)

    elif args.action == 'predict':
        # Demo prediction
        predictor = XGBoostPredictor()
        test_processes = [
            {'process_type': 'cpu_bound', 'priority': 15, 'io_frequency': 0.1,
             'memory_mb': 1024, 'num_threads': 8},
            {'process_type': 'io_bound', 'priority': 3, 'io_frequency': 0.8,
             'memory_mb': 64, 'num_threads': 1},
            {'process_type': 'interactive', 'priority': 1, 'io_frequency': 0.5,
             'memory_mb': 256, 'num_threads': 2},
        ]
        print("\nSample Predictions:")
        for p in test_processes:
            burst = predictor.predict(p)
            print(f"  {p['process_type']:<12} → predicted burst: {burst:.1f}")

    elif args.action == 'stats':
        stats = get_model_stats()
        print(json.dumps(stats, indent=2))
