"""
mock_data.py — Mock Data for Independent Dashboard Development
================================================================
Provides realistic fake benchmark results and model stats so the
dashboard can be developed and tested without Modules 1 and 2.

Set USE_MOCK = True in app.py to use this data.
Set USE_MOCK = False when Modules 1 and 2 are integrated.
"""

import random


def get_mock_benchmark_results():
    """
    Returns mock benchmark results mimicking what the real
    benchmark.py would produce.

    The values are realistic:
    - SJF has the lowest waiting time (it's theoretically optimal)
    - FCFS has the highest waiting time (convoy effect)
    - ML schedulers sit between FCFS and SJF
    - Round Robin has good response time but higher turnaround
    """
    return {
        "metrics": [
            {
                "algorithm": "FCFS",
                "avg_waiting_time": 142.53,
                "avg_turnaround_time": 165.28,
                "avg_response_time": 142.53,
                "cpu_utilization": 97.8,
                "throughput": 0.0412,
                "fairness_index": 0.7823,
                "max_waiting_time": 890,
                "total_time": 1215,
            },
            {
                "algorithm": "SJF (Known Burst)",
                "avg_waiting_time": 68.41,
                "avg_turnaround_time": 91.16,
                "avg_response_time": 68.41,
                "cpu_utilization": 98.2,
                "throughput": 0.0421,
                "fairness_index": 0.7156,
                "max_waiting_time": 1105,
                "total_time": 1190,
            },
            {
                "algorithm": "Priority",
                "avg_waiting_time": 118.72,
                "avg_turnaround_time": 141.47,
                "avg_response_time": 118.72,
                "cpu_utilization": 97.9,
                "throughput": 0.0415,
                "fairness_index": 0.7534,
                "max_waiting_time": 945,
                "total_time": 1205,
            },
            {
                "algorithm": "Round Robin (q=4)",
                "avg_waiting_time": 135.89,
                "avg_turnaround_time": 158.64,
                "avg_response_time": 12.34,
                "cpu_utilization": 97.5,
                "throughput": 0.0408,
                "fairness_index": 0.9412,
                "max_waiting_time": 420,
                "total_time": 1225,
            },
            {
                "algorithm": "XGBoost-SJF",
                "avg_waiting_time": 82.15,
                "avg_turnaround_time": 104.90,
                "avg_response_time": 82.15,
                "cpu_utilization": 98.1,
                "throughput": 0.0419,
                "fairness_index": 0.7298,
                "max_waiting_time": 1020,
                "total_time": 1195,
            },
            {
                "algorithm": "LSTM-SJF",
                "avg_waiting_time": 76.33,
                "avg_turnaround_time": 99.08,
                "avg_response_time": 76.33,
                "cpu_utilization": 98.1,
                "throughput": 0.0420,
                "fairness_index": 0.7201,
                "max_waiting_time": 1050,
                "total_time": 1192,
            },
            {
                "algorithm": "DQN Scheduler",
                "avg_waiting_time": 95.67,
                "avg_turnaround_time": 118.42,
                "avg_response_time": 28.91,
                "cpu_utilization": 97.9,
                "throughput": 0.0416,
                "fairness_index": 0.8845,
                "max_waiting_time": 510,
                "total_time": 1202,
            },
        ],
        "gantt": _generate_mock_gantt(),
        "processes": _generate_mock_processes(),
    }


def get_mock_model_stats():
    """Returns mock ML model performance metrics."""
    return {
        "xgboost": {
            "mae": 3.21,
            "rmse": 4.56,
            "r2": 0.87,
            "feature_importances": {
                "process_type_cpu_bound": 0.32,
                "io_frequency": 0.28,
                "memory_mb": 0.15,
                "num_threads": 0.12,
                "priority": 0.08,
                "process_type_io_bound": 0.03,
                "process_type_mixed": 0.01,
                "process_type_interactive": 0.01,
            },
        },
        "lstm": {
            "mae": 2.89,
            "rmse": 4.12,
            "r2": 0.91,
            "training_loss": [411.6, 280.3, 225.1, 210.5, 197.7, 196.8, 195.4, 194.1, 193.5, 192.9],
            "epochs": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
        },
        "dqn": {
            "avg_reward": -45.2,
            "episodes_trained": 5000,
            "reward_history": [-85.3, -62.1, -48.7, -39.2, -35.8, -31.4, -28.9, -25.1, -22.6, -19.1],
            "episode_checkpoints": [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
        },
    }


def _generate_mock_gantt():
    """Generate a small mock Gantt chart for the first 30 time units, per algorithm."""
    random.seed(99)
    gantt = {}
    algorithms = [
        "FCFS", "SJF (Known Burst)", "Priority",
        "Round Robin (q=4)", "XGBoost-SJF", "LSTM-SJF", "DQN Scheduler",
    ]
    for algo in algorithms:
        log = []
        pids = list(range(1, 8))
        random.shuffle(pids)
        t = 0
        for pid in pids:
            burst = random.randint(2, 6)
            for _ in range(burst):
                log.append({"time": t, "pid": pid})
                t += 1
        gantt[algo] = log[:30]  # Only first 30 ticks for UI preview
    return gantt


def _generate_mock_processes():
    """Generate mock process list with predicted burst times."""
    random.seed(42)
    types = ['cpu_bound', 'io_bound', 'mixed', 'interactive']
    processes = []
    for pid in range(1, 21):
        ptype = random.choice(types)
        burst = random.randint(3, 40)
        processes.append({
            "pid": pid,
            "process_type": ptype,
            "priority": random.randint(1, 20),
            "arrival_time": random.randint(0, 50),
            "cpu_burst": burst,
            "predicted_burst_xgb": round(burst + random.uniform(-4, 4), 1),
            "predicted_burst_lstm": round(burst + random.uniform(-3, 3), 1),
        })
    return processes