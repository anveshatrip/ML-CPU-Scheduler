"""
lstm_model.py — LSTM Burst Time Predictor (Sequential)
========================================================
WHAT THIS DOES:
    Trains an LSTM (Long Short-Term Memory) neural network to predict CPU
    burst times by looking at SEQUENCES of recent processes.

WHY LSTM (vs XGBoost):
    XGBoost looks at each process INDEPENDENTLY — it sees one process's
    features and predicts its burst. It has no concept of "what came before."

    LSTM looks at a SEQUENCE of processes (e.g., the last 5 processes)
    and predicts the next burst. It can learn temporal patterns like:
    - "After 3 I/O-bound processes, a CPU-heavy one usually comes"
    - "Interactive processes tend to arrive in bursts during work hours"
    - "CPU-bound processes often follow other CPU-bound processes"

    Think of it like weather forecasting: knowing yesterday's temperature
    helps predict today's, even if both days have similar humidity.

HOW LSTM WORKS (simplified):
    An LSTM cell has a "memory" that it carries forward through the sequence.
    At each step, it decides:
    1. What to FORGET from previous memory (forget gate)
    2. What NEW information to STORE (input gate)
    3. What to OUTPUT as the prediction (output gate)

    This lets it capture long-range dependencies — patterns that span
    many time steps — which simpler models can't do.

ARCHITECTURE:
    Input (sequence of 5 processes × 8 features)
    → LSTM layer (64 hidden units, 2 layers)
    → Dropout (0.2, prevents overfitting)
    → Fully Connected layer (64 → 32)
    → ReLU activation
    → Fully Connected layer (32 → 1)
    → Output (predicted burst time)

USAGE:
    python -m module2_ml.lstm_model train
    python -m module2_ml.lstm_model predict
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'saved_models')
MODEL_PATH = os.path.join(MODEL_DIR, 'lstm_model.pt')
SCALER_PATH = os.path.join(MODEL_DIR, 'lstm_scaler.pkl')
STATS_PATH = os.path.join(MODEL_DIR, 'lstm_stats.json')

# How many previous processes the LSTM looks at to predict the next one
# Think of this as "window size" — the model sees the last 5 processes
SEQUENCE_LENGTH = 5

# Features used (same as XGBoost, but order matters for the scaler)
FEATURE_COLUMNS = ['priority', 'io_frequency', 'memory_mb', 'num_threads']
PROCESS_TYPES = ['cpu_bound', 'io_bound', 'mixed', 'interactive']

# Training hyperparameters
LSTM_CONFIG = {
    'hidden_size': 64,      # Number of LSTM hidden units
    'num_layers': 2,        # Stacked LSTM layers (deeper = more capacity)
    'dropout': 0.2,         # Dropout rate (prevents overfitting)
    'learning_rate': 0.001, # Adam optimizer learning rate
    'batch_size': 64,       # Processes per training batch
    'epochs': 50,           # Number of training passes over the data
}


# ═══════════════════════════════════════════════════════════════════════
# DATASET — Convert process data into sequences
# ═══════════════════════════════════════════════════════════════════════
class ProcessSequenceDataset(Dataset):
    """
    Converts a flat list of processes into overlapping sequences.

    EXAMPLE (SEQUENCE_LENGTH=3):
        Processes: [P1, P2, P3, P4, P5]
        Becomes:
            Sequence [P1, P2, P3] → Target: P4's burst
            Sequence [P2, P3, P4] → Target: P5's burst

    Each process is represented by its 8 features (4 numerical + 4 one-hot).
    The target is the cpu_burst of the process AFTER the sequence.

    WHY SEQUENCES?
    This is what makes LSTM different from XGBoost. Instead of predicting
    burst from a single process's features, we predict from the CONTEXT
    of what processes ran recently. The LSTM learns patterns in the
    sequence itself.
    """

    def __init__(self, features: np.ndarray, targets: np.ndarray, seq_len: int):
        """
        Args:
            features: 2D array of shape (n_processes, n_features)
            targets: 1D array of cpu_burst values
            seq_len: Number of processes in each input sequence
        """
        self.sequences = []
        self.labels = []

        # Create overlapping sequences with a sliding window
        for i in range(len(features) - seq_len):
            # Input: features of processes [i, i+1, ..., i+seq_len-1]
            seq = features[i:i + seq_len]
            # Target: burst time of process [i+seq_len]
            label = targets[i + seq_len]
            self.sequences.append(torch.FloatTensor(seq))
            self.labels.append(torch.FloatTensor([label]))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


# ═══════════════════════════════════════════════════════════════════════
# LSTM NEURAL NETWORK
# ═══════════════════════════════════════════════════════════════════════
class BurstPredictorLSTM(nn.Module):
    """
    LSTM neural network for burst time prediction.

    ARCHITECTURE:
        Input: (batch_size, sequence_length, num_features)
        ↓
        LSTM layers (captures sequential patterns)
        ↓
        Dropout (prevents overfitting)
        ↓
        FC layer 1: hidden_size → 32 (compression)
        ↓
        ReLU activation (non-linearity)
        ↓
        FC layer 2: 32 → 1 (final prediction)
        ↓
        Output: predicted burst time (scalar)
    """

    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,    # Number of features per process
            hidden_size=hidden_size,  # LSTM internal memory size
            num_layers=num_layers,    # Stack multiple LSTM layers
            batch_first=True,         # Input shape: (batch, seq, features)
            dropout=dropout if num_layers > 1 else 0,  # Between LSTM layers
        )

        self.dropout = nn.Dropout(dropout)

        # Fully connected layers to convert LSTM output to burst prediction
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, num_features)

        Returns:
            Predicted burst times of shape (batch_size, 1)
        """
        # LSTM processes the sequence and outputs hidden states
        # lstm_out shape: (batch_size, sequence_length, hidden_size)
        # We only care about the LAST time step's output
        lstm_out, (hidden, cell) = self.lstm(x)

        # Take the output from the last time step
        last_output = lstm_out[:, -1, :]  # shape: (batch_size, hidden_size)

        # Apply dropout and fully connected layers
        out = self.dropout(last_output)
        prediction = self.fc(out)  # shape: (batch_size, 1)

        return prediction


# ═══════════════════════════════════════════════════════════════════════
# FEATURE PREPARATION
# ═══════════════════════════════════════════════════════════════════════
def prepare_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Prepare features for LSTM training.

    Unlike XGBoost, LSTM benefits from SCALED features (values between 0 and 1).
    This is because neural networks use gradient descent, and features with
    very different scales (e.g., memory_mb: 10-2048 vs io_frequency: 0-1)
    can cause training instability.

    Returns:
        features: Scaled 2D array (n_processes, n_features)
        targets: 1D array of burst times
        scaler: Fitted StandardScaler (needed for prediction)
    """
    # One-hot encode process_type
    for ptype in PROCESS_TYPES:
        df[f'process_type_{ptype}'] = (df['process_type'] == ptype).astype(int)

    feature_cols = FEATURE_COLUMNS + [f'process_type_{t}' for t in PROCESS_TYPES]
    features = df[feature_cols].values.astype(np.float32)
    targets = df['cpu_burst'].values.astype(np.float32)

    # Scale features to zero mean, unit variance
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    return features, targets, scaler


# ═══════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════
def train(data_path: str = 'data/processes.csv'):
    """
    Train the LSTM model.

    STEP BY STEP:
    1. Load and prepare data
    2. Create sequence datasets
    3. Build LSTM model
    4. Train with Adam optimizer + MSE loss
    5. Evaluate on test set
    6. Save model, scaler, and stats
    """
    print("=" * 60)
    print("  LSTM Burst Time Predictor — Training")
    print("=" * 60)

    # Detect device (CPU vs GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Device: {device}")

    # Step 1: Load and prepare data
    print(f"\n[1/5] Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    features, targets, scaler = prepare_features(df.copy())
    num_features = features.shape[1]
    print(f"      {len(df)} processes, {num_features} features")

    # Step 2: Create train/test sequence datasets
    print(f"[2/5] Creating sequences (window={SEQUENCE_LENGTH})...")
    split_idx = int(len(features) * 0.8)

    train_dataset = ProcessSequenceDataset(
        features[:split_idx], targets[:split_idx], SEQUENCE_LENGTH
    )
    test_dataset = ProcessSequenceDataset(
        features[split_idx:], targets[split_idx:], SEQUENCE_LENGTH
    )
    print(f"      Train: {len(train_dataset)} sequences")
    print(f"      Test:  {len(test_dataset)} sequences")

    train_loader = DataLoader(train_dataset, batch_size=LSTM_CONFIG['batch_size'],
                              shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=LSTM_CONFIG['batch_size'])

    # Step 3: Build model
    print("[3/5] Building LSTM model...")
    model = BurstPredictorLSTM(
        input_size=num_features,
        hidden_size=LSTM_CONFIG['hidden_size'],
        num_layers=LSTM_CONFIG['num_layers'],
        dropout=LSTM_CONFIG['dropout'],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"      Architecture: LSTM({num_features}→{LSTM_CONFIG['hidden_size']}) → FC(32→1)")
    print(f"      Total parameters: {total_params:,}")

    # Loss function and optimizer
    criterion = nn.MSELoss()  # Mean Squared Error — standard for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=LSTM_CONFIG['learning_rate'])

    # Step 4: Training loop
    print(f"[4/5] Training for {LSTM_CONFIG['epochs']} epochs...")
    best_loss = float('inf')

    for epoch in range(LSTM_CONFIG['epochs']):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            # Forward pass
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)

            # Backward pass (gradient descent)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"      Epoch {epoch+1:>3}/{LSTM_CONFIG['epochs']} — "
                  f"Train Loss: {avg_train_loss:.4f}")

        # Save best model
        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            torch.save(model.state_dict(), MODEL_PATH)

    # Step 5: Evaluate on test set
    print("[5/5] Evaluating on test set...")
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            preds = model(batch_X).cpu().numpy().flatten()
            all_preds.extend(preds)
            all_targets.extend(batch_y.numpy().flatten())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    mae = mean_absolute_error(all_targets, all_preds)
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    r2 = r2_score(all_targets, all_preds)

    print(f"\n      ┌────────────────────────────┐")
    print(f"      │  MAE  (Mean Abs Error): {mae:>6.2f} │")
    print(f"      │  RMSE (Root MSE):       {rmse:>6.2f} │")
    print(f"      │  R² Score:              {r2:>6.3f} │")
    print(f"      └────────────────────────────┘")

    # Save scaler (needed for prediction)
    import joblib
    joblib.dump(scaler, SCALER_PATH)

    # Save stats
    os.makedirs(MODEL_DIR, exist_ok=True)
    stats = {
        'mae': round(float(mae), 4),
        'rmse': round(float(rmse), 4),
        'r2': round(float(r2), 4),
        'sequence_length': SEQUENCE_LENGTH,
        'total_params': total_params,
        'epochs': LSTM_CONFIG['epochs'],
        'device': str(device),
    }
    with open(STATS_PATH, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n[✓] Model saved to {MODEL_PATH}")
    print(f"[✓] Scaler saved to {SCALER_PATH}")
    print(f"[✓] Stats saved to {STATS_PATH}")

    return stats


# ═══════════════════════════════════════════════════════════════════════
# PREDICTION
# ═══════════════════════════════════════════════════════════════════════
class LSTMPredictor:
    """
    Wrapper for making burst time predictions with the trained LSTM.

    KEY DIFFERENCE from XGBoost:
    The LSTM needs a SEQUENCE of recent processes, not just one.
    You must call add_to_history() for each process as it completes,
    and predict() uses the last SEQUENCE_LENGTH processes as context.

    USAGE:
        predictor = LSTMPredictor()

        # Feed in recent process history
        for recent_process in last_5_processes:
            predictor.add_to_history(recent_process)

        # Predict burst for new process
        predicted_burst = predictor.predict(new_process_dict)
    """

    def __init__(self, model_path: str = MODEL_PATH, scaler_path: str = SCALER_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                f"Run 'python -m module2_ml.lstm_model train' first."
            )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load scaler
        import joblib
        self.scaler = joblib.load(scaler_path)

        # Determine model input size from scaler
        self.num_features = len(FEATURE_COLUMNS) + len(PROCESS_TYPES)

        # Load model
        self.model = BurstPredictorLSTM(
            input_size=self.num_features,
            hidden_size=LSTM_CONFIG['hidden_size'],
            num_layers=LSTM_CONFIG['num_layers'],
            dropout=0,  # No dropout during inference
        ).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device,
                                               weights_only=True))
        self.model.eval()

        # History buffer — stores recent process features
        self.history = []

    def _process_to_features(self, process_dict: dict) -> np.ndarray:
        """Convert a process dict to a feature vector."""
        features = [
            process_dict['priority'],
            process_dict['io_frequency'],
            process_dict['memory_mb'],
            process_dict['num_threads'],
        ]
        for ptype in PROCESS_TYPES:
            features.append(1 if process_dict['process_type'] == ptype else 0)
        return np.array(features, dtype=np.float32)

    def add_to_history(self, process_dict: dict):
        """
        Add a completed process to the history buffer.
        Keeps only the last SEQUENCE_LENGTH processes.
        """
        features = self._process_to_features(process_dict)
        self.history.append(features)
        if len(self.history) > SEQUENCE_LENGTH:
            self.history = self.history[-SEQUENCE_LENGTH:]

    def predict(self, process_dict: dict) -> float:
        """
        Predict burst time using the current history + new process features.

        If history has fewer than SEQUENCE_LENGTH entries, pads with zeros.

        Returns:
            Predicted burst time (float, always >= 1)
        """
        current_features = self._process_to_features(process_dict)

        # Build the input sequence
        sequence = list(self.history)
        # Pad if not enough history
        while len(sequence) < SEQUENCE_LENGTH - 1:
            sequence.insert(0, np.zeros(self.num_features, dtype=np.float32))
        # Add the current process as the last in the sequence
        sequence.append(current_features)
        # Take only the last SEQUENCE_LENGTH
        sequence = sequence[-SEQUENCE_LENGTH:]

        # Scale features
        seq_array = np.array(sequence)
        seq_scaled = self.scaler.transform(seq_array)

        # Convert to tensor and predict
        input_tensor = torch.FloatTensor(seq_scaled).unsqueeze(0).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_tensor).item()

        return max(1.0, prediction)

    def reset_history(self):
        """Clear the history buffer (e.g., between benchmark runs)."""
        self.history = []


# ═══════════════════════════════════════════════════════════════════════
# MODEL STATS (for dashboard)
# ═══════════════════════════════════════════════════════════════════════
def get_model_stats() -> dict:
    """Load and return saved model stats."""
    if os.path.exists(STATS_PATH):
        with open(STATS_PATH, 'r') as f:
            return json.load(f)
    return {'error': 'Model not trained yet.'}


# ═══════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='LSTM Burst Time Predictor')
    parser.add_argument('action', choices=['train', 'predict', 'stats'])
    parser.add_argument('--data', type=str, default='data/processes.csv')
    args = parser.parse_args()

    if args.action == 'train':
        train(data_path=args.data)

    elif args.action == 'predict':
        predictor = LSTMPredictor()
        test_processes = [
            {'process_type': 'io_bound', 'priority': 2, 'io_frequency': 0.9,
             'memory_mb': 32, 'num_threads': 1},
            {'process_type': 'mixed', 'priority': 10, 'io_frequency': 0.4,
             'memory_mb': 512, 'num_threads': 4},
            {'process_type': 'cpu_bound', 'priority': 18, 'io_frequency': 0.1,
             'memory_mb': 1500, 'num_threads': 12},
        ]
        print("\nSequential Predictions (building up history):")
        for p in test_processes:
            burst = predictor.predict(p)
            print(f"  {p['process_type']:<12} → predicted burst: {burst:.1f}")
            predictor.add_to_history(p)

    elif args.action == 'stats':
        print(json.dumps(get_model_stats(), indent=2))
