"""
dqn_agent.py — Deep Q-Network Reinforcement Learning Scheduler
================================================================
WHAT THIS DOES:
    Instead of predicting burst times (like XGBoost/LSTM), the DQN agent
    learns a SCHEDULING POLICY directly through trial and error.

    It observes the state of the ready queue, picks a process to run,
    and receives a reward. Over thousands of episodes, it learns which
    process to pick in which situation to minimize total waiting time.

WHY REINFORCEMENT LEARNING:
    Supervised learning (XGBoost, LSTM) learns from labeled examples.
    RL learns from EXPERIENCE — like how a chess AI learns by playing
    millions of games, not by studying labeled board positions.

    The beauty of RL for scheduling: it can discover strategies that
    no human-designed algorithm uses. It might learn to batch similar
    process types, or to strategically delay certain processes.

HOW DQN WORKS:
    DQN (Deep Q-Network) is a popular RL algorithm introduced by DeepMind.

    Core concepts:
    1. STATE:   What the agent sees (ready queue stats, process features)
    2. ACTION:  Which process to schedule next
    3. REWARD:  Negative of waiting time increase (we want to MINIMIZE wait)
    4. Q-VALUE: The expected total future reward for taking an action in a state
    5. NETWORK: A neural net that predicts Q-values for all possible actions

    The agent learns to pick the action with the highest Q-value in each state.

    KEY TRICK: Experience Replay
    Instead of learning from the most recent experience only, the agent
    stores past experiences in a "replay buffer" and randomly samples
    from it. This breaks correlations and stabilizes training.

ARCHITECTURE:
    State (14 features)
    → FC layer (14 → 128) + ReLU
    → FC layer (128 → 128) + ReLU
    → FC layer (128 → max_queue_size)
    → Output: Q-value for each possible action (= each position in queue)

USAGE:
    python -m module2_ml.dqn_agent train
    python -m module2_ml.dqn_agent stats
"""

import os
import json
import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim


# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'saved_models')
MODEL_PATH = os.path.join(MODEL_DIR, 'dqn_model.pt')
STATS_PATH = os.path.join(MODEL_DIR, 'dqn_stats.json')

# DQN Hyperparameters
DQN_CONFIG = {
    'state_size': 14,           # Size of the state vector (see _get_state)
    'max_queue_size': 20,       # Max processes in ready queue at once
    'hidden_size': 128,         # Hidden layer size in the Q-network
    'learning_rate': 0.001,     # Adam optimizer learning rate
    'gamma': 0.99,              # Discount factor (how much to value future rewards)
    'epsilon_start': 1.0,       # Initial exploration rate (100% random at start)
    'epsilon_end': 0.05,        # Minimum exploration rate
    'epsilon_decay': 0.995,     # How fast exploration decreases per episode
    'replay_buffer_size': 10000,# How many experiences to store
    'batch_size': 64,           # Experiences per training batch
    'target_update_freq': 10,   # Update target network every N episodes
    'episodes': 2000,           # Total training episodes
}


# ═══════════════════════════════════════════════════════════════════════
# Q-NETWORK (the "brain" of the agent)
# ═══════════════════════════════════════════════════════════════════════
class QNetwork(nn.Module):
    """
    Neural network that estimates Q-values.

    Q-VALUE EXPLAINED:
    Q(state, action) = "How good is it to take this action in this state?"
    A high Q-value means the agent expects good future outcomes.
    The agent always picks the action with the highest Q-value (after training).

    ARCHITECTURE:
    Simple 3-layer MLP (Multi-Layer Perceptron):
        state (14) → 128 → ReLU → 128 → ReLU → max_queue_size
    """

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, state):
        return self.network(state)


# ═══════════════════════════════════════════════════════════════════════
# REPLAY BUFFER (stores past experiences for learning)
# ═══════════════════════════════════════════════════════════════════════
class ReplayBuffer:
    """
    Stores (state, action, reward, next_state, done) tuples.

    WHY REPLAY BUFFER?
    If the agent only learned from the most recent experience, it would
    quickly forget earlier lessons. The replay buffer stores thousands
    of experiences and randomly samples batches for training.

    Benefits:
    - Breaks correlation between consecutive experiences
    - Reuses rare but important experiences multiple times
    - Stabilizes training significantly
    """

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones),
        )

    def __len__(self):
        return len(self.buffer)


# ═══════════════════════════════════════════════════════════════════════
# SCHEDULING ENVIRONMENT (simulates the CPU scheduling problem)
# ═══════════════════════════════════════════════════════════════════════
class SchedulingEnv:
    """
    A simplified scheduling environment for RL training.

    The agent interacts with this environment:
    1. Observes the state (ready queue stats)
    2. Takes an action (picks a process from the queue)
    3. Receives a reward (negative waiting time impact)
    4. Transitions to the next state

    STATE REPRESENTATION (14 features):
    The state vector encodes the current situation:
    - Queue size (normalized)
    - Average burst of queued processes
    - Average priority of queued processes
    - Average I/O frequency
    - Time elapsed (normalized)
    - Stats about the top 3 processes (burst, priority, I/O for each)
    """

    def __init__(self, processes: list, max_queue_size: int = 20):
        self.all_processes = processes
        self.max_queue_size = max_queue_size
        self.reset()

    def reset(self):
        """Reset environment to initial state."""
        # Deep copy process data as dicts
        self.remaining = [p.copy() for p in self.all_processes]
        self.remaining.sort(key=lambda p: p['arrival_time'])
        self.ready_queue = []
        self.current_time = 0
        self.total_waiting = 0
        self.completed = 0
        self.next_arrival_idx = 0
        self._advance_arrivals()
        return self._get_state()

    def _advance_arrivals(self):
        """Move processes that have arrived into the ready queue."""
        while (self.next_arrival_idx < len(self.remaining) and
               self.remaining[self.next_arrival_idx]['arrival_time'] <= self.current_time):
            if len(self.ready_queue) < self.max_queue_size:
                self.ready_queue.append(self.remaining[self.next_arrival_idx])
            self.next_arrival_idx += 1

    def _get_state(self) -> np.ndarray:
        """
        Encode the current scheduling situation as a fixed-size vector.

        This is what the agent "sees" — it must contain enough information
        for the agent to make a good scheduling decision.
        """
        state = np.zeros(DQN_CONFIG['state_size'], dtype=np.float32)

        if not self.ready_queue:
            return state

        q = self.ready_queue

        # Global features
        state[0] = len(q) / self.max_queue_size                      # Queue fill %
        state[1] = np.mean([p['cpu_burst'] for p in q]) / 50.0       # Avg burst (norm)
        state[2] = np.mean([p['priority'] for p in q]) / 20.0        # Avg priority (norm)
        state[3] = np.mean([p['io_frequency'] for p in q])           # Avg I/O freq
        state[4] = self.current_time / 1000.0                        # Time elapsed (norm)

        # Top 3 candidate features (sorted by burst time)
        sorted_q = sorted(q, key=lambda p: p['cpu_burst'])
        for i in range(min(3, len(sorted_q))):
            base = 5 + i * 3
            state[base] = sorted_q[i]['cpu_burst'] / 50.0            # Burst (norm)
            state[base + 1] = sorted_q[i]['priority'] / 20.0         # Priority (norm)
            state[base + 2] = sorted_q[i]['io_frequency']            # I/O freq

        return state

    def step(self, action: int) -> tuple:
        """
        Execute the agent's action (schedule process at index 'action').

        Args:
            action: Index into the ready queue (which process to run)

        Returns:
            (next_state, reward, done, info)
            - next_state: New state after the action
            - reward: Negative waiting time (we want to minimize this)
            - done: True if all processes are complete
            - info: Extra info dict
        """
        if not self.ready_queue:
            return self._get_state(), 0, True, {}

        # Clamp action to valid range
        action = min(action, len(self.ready_queue) - 1)

        # Pick the chosen process
        chosen = self.ready_queue.pop(action)
        burst = chosen['cpu_burst']

        # All OTHER processes in the queue accumulate waiting time
        waiting_penalty = burst * len(self.ready_queue)
        self.total_waiting += waiting_penalty

        # Advance time
        self.current_time += burst
        self.completed += 1

        # Check for new arrivals
        self._advance_arrivals()

        # Calculate reward: negative of waiting time caused by this decision
        # The agent learns to pick actions that cause the LEAST waiting
        reward = -waiting_penalty / max(1, len(self.all_processes))

        # Check if done
        done = (self.completed >= len(self.all_processes) or
                (not self.ready_queue and self.next_arrival_idx >= len(self.remaining)))

        # If queue is empty but processes remain, fast-forward
        if not self.ready_queue and not done and self.next_arrival_idx < len(self.remaining):
            self.current_time = self.remaining[self.next_arrival_idx]['arrival_time']
            self._advance_arrivals()

        return self._get_state(), reward, done, {'waiting': waiting_penalty}

    @property
    def num_actions(self):
        """Number of valid actions = number of processes in ready queue."""
        return len(self.ready_queue)


# ═══════════════════════════════════════════════════════════════════════
# DQN AGENT
# ═══════════════════════════════════════════════════════════════════════
class DQNAgent:
    """
    The DQN agent that learns to schedule processes.

    KEY COMPONENTS:
    1. policy_net:  The main Q-network that is actively trained
    2. target_net:  A slowly-updated copy used for stable Q-value targets
    3. replay_buffer: Stores past experiences for batch learning
    4. epsilon: Controls exploration vs exploitation

    EPSILON-GREEDY EXPLORATION:
    Early in training, epsilon is high (1.0) → agent takes random actions
    (exploring different strategies). Over time, epsilon decreases → agent
    relies more on its learned Q-values (exploiting what it's learned).
    """

    def __init__(self, state_size: int = DQN_CONFIG['state_size'],
                 action_size: int = DQN_CONFIG['max_queue_size']):

        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Two networks: policy (actively trained) and target (stable reference)
        self.policy_net = QNetwork(state_size, action_size,
                                   DQN_CONFIG['hidden_size']).to(self.device)
        self.target_net = QNetwork(state_size, action_size,
                                   DQN_CONFIG['hidden_size']).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target net is never trained directly

        self.optimizer = optim.Adam(self.policy_net.parameters(),
                                    lr=DQN_CONFIG['learning_rate'])
        self.replay_buffer = ReplayBuffer(DQN_CONFIG['replay_buffer_size'])
        self.epsilon = DQN_CONFIG['epsilon_start']

    def select_action(self, state: np.ndarray, num_valid_actions: int) -> int:
        """
        Choose which process to schedule.

        Uses epsilon-greedy: with probability epsilon, pick randomly.
        Otherwise, pick the action with the highest Q-value.

        Args:
            state: Current environment state
            num_valid_actions: Number of processes in the ready queue

        Returns:
            Index of the chosen process in the ready queue
        """
        if num_valid_actions == 0:
            return 0

        if random.random() < self.epsilon:
            # EXPLORE: pick a random process
            return random.randint(0, num_valid_actions - 1)
        else:
            # EXPLOIT: pick the process the model thinks is best
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                # Only consider valid actions (processes actually in queue)
                valid_q = q_values[0, :num_valid_actions]
                return valid_q.argmax().item()

    def learn(self):
        """
        Sample a batch from the replay buffer and update the Q-network.

        THE DQN LEARNING FORMULA:
            target = reward + gamma * max(Q_target(next_state, all_actions))
            loss = MSE(Q_policy(state, action), target)

        In English:
            "The Q-value for this (state, action) should equal the reward
            I got, plus the discounted best Q-value of the next state."
        """
        if len(self.replay_buffer) < DQN_CONFIG['batch_size']:
            return  # Not enough experiences yet

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            DQN_CONFIG['batch_size']
        )
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Current Q-values for the actions we took
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Target Q-values (what we SHOULD have predicted)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + DQN_CONFIG['gamma'] * next_q * (1 - dones)

        # Loss and backpropagation
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        """Copy policy network weights to target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        """Reduce exploration rate."""
        self.epsilon = max(DQN_CONFIG['epsilon_end'],
                          self.epsilon * DQN_CONFIG['epsilon_decay'])

    def save(self, path: str = MODEL_PATH):
        """Save the trained policy network."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path: str = MODEL_PATH):
        """Load a trained policy network."""
        self.policy_net.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True)
        )
        self.policy_net.eval()
        self.epsilon = 0  # No exploration during inference


# ═══════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════
def train(data_path: str = 'data/processes.csv'):
    """
    Train the DQN agent through simulated scheduling episodes.

    Each episode:
    1. Generate a random subset of processes
    2. Let the agent schedule them (making decisions, receiving rewards)
    3. Store experiences in replay buffer
    4. Learn from sampled batch
    5. Decay exploration rate
    """
    import pandas as pd

    print("=" * 60)
    print("  DQN Agent — Reinforcement Learning Training")
    print("=" * 60)

    # Load process data
    print(f"\n[1/3] Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    all_processes = df.to_dict('records')
    print(f"      Loaded {len(all_processes)} processes")

    # Create agent
    print("[2/3] Creating DQN agent...")
    agent = DQNAgent()
    print(f"      State size: {agent.state_size}")
    print(f"      Action size: {agent.action_size}")
    print(f"      Device: {agent.device}")

    # Training loop
    print(f"[3/3] Training for {DQN_CONFIG['episodes']} episodes...")
    rewards_history = []
    best_avg_reward = -float('inf')

    for episode in range(DQN_CONFIG['episodes']):
        # Randomly sample 30-80 processes for this episode
        num_procs = random.randint(30, min(80, len(all_processes)))
        episode_processes = random.sample(all_processes, num_procs)

        # Create environment and reset
        env = SchedulingEnv(episode_processes)
        state = env.reset()
        episode_reward = 0

        # Run the episode
        while True:
            num_actions = env.num_actions
            if num_actions == 0:
                break

            # Agent picks an action
            action = agent.select_action(state, num_actions)

            # Environment responds
            next_state, reward, done, info = env.step(action)

            # Store experience
            agent.replay_buffer.push(state, action, reward, next_state, float(done))

            # Learn from experience
            agent.learn()

            episode_reward += reward
            state = next_state

            if done:
                break

        rewards_history.append(episode_reward)
        agent.decay_epsilon()

        # Update target network periodically
        if (episode + 1) % DQN_CONFIG['target_update_freq'] == 0:
            agent.update_target_network()

        # Print progress
        if (episode + 1) % 200 == 0:
            avg_reward = np.mean(rewards_history[-200:])
            print(f"      Episode {episode+1:>5}/{DQN_CONFIG['episodes']} | "
                  f"Avg Reward: {avg_reward:>8.2f} | "
                  f"Epsilon: {agent.epsilon:.3f}")

            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                agent.save()

    # Final save
    agent.save()

    # Save stats
    final_avg_reward = np.mean(rewards_history[-200:])
    stats = {
        'avg_reward': round(float(final_avg_reward), 2),
        'episodes_trained': DQN_CONFIG['episodes'],
        'best_avg_reward': round(float(best_avg_reward), 2),
        'epsilon_final': round(agent.epsilon, 4),
        'config': DQN_CONFIG,
    }
    with open(STATS_PATH, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n      ┌──────────────────────────────────┐")
    print(f"      │  Final Avg Reward:   {final_avg_reward:>10.2f}  │")
    print(f"      │  Best Avg Reward:    {best_avg_reward:>10.2f}  │")
    print(f"      │  Final Epsilon:      {agent.epsilon:>10.4f}  │")
    print(f"      └──────────────────────────────────┘")
    print(f"\n[✓] Model saved to {MODEL_PATH}")
    print(f"[✓] Stats saved to {STATS_PATH}")

    return stats


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
    parser = argparse.ArgumentParser(description='DQN Scheduling Agent')
    parser.add_argument('action', choices=['train', 'stats'])
    parser.add_argument('--data', type=str, default='data/processes.csv')
    args = parser.parse_args()

    if args.action == 'train':
        train(data_path=args.data)
    elif args.action == 'stats':
        print(json.dumps(get_model_stats(), indent=2))
