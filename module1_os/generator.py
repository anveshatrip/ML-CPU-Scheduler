import os
import random
import numpy as np
import pandas as pd
from module1_os.process import Process

PROCESS_PROFILES = {
    'cpu_bound': {
        'cpu_burst_range': (20, 50), 'io_freq_range': (0.0, 0.2),
        'memory_range': (256, 2048), 'thread_range': (2, 16),
        'priority_range': (8, 20), 'weight': 0.25,
    },
    'io_bound': {
        'cpu_burst_range': (1, 8), 'io_freq_range': (0.6, 1.0),
        'memory_range': (10, 256), 'thread_range': (1, 4),
        'priority_range': (1, 10), 'weight': 0.30,
    },
    'mixed': {
        'cpu_burst_range': (8, 25), 'io_freq_range': (0.3, 0.6),
        'memory_range': (128, 1024), 'thread_range': (1, 8),
        'priority_range': (5, 15), 'weight': 0.30,
    },
    'interactive': {
        'cpu_burst_range': (1, 5), 'io_freq_range': (0.4, 0.8),
        'memory_range': (50, 512), 'thread_range': (1, 4),
        'priority_range': (1, 5), 'weight': 0.15,
    },
}


def generate_process(pid, process_type, arrival_time):
    profile = PROCESS_PROFILES[process_type]
    base_burst = random.randint(*profile['cpu_burst_range'])
    burst_range_width = profile['cpu_burst_range'][1] - profile['cpu_burst_range'][0]
    noise = np.random.normal(0, burst_range_width * 0.15)
    cpu_burst = max(1, int(base_burst + noise))
    return Process(
        pid=pid, process_type=process_type,
        priority=random.randint(*profile['priority_range']),
        arrival_time=arrival_time, cpu_burst=cpu_burst,
        io_frequency=round(random.uniform(*profile['io_freq_range']), 2),
        memory_mb=random.randint(*profile['memory_range']),
        num_threads=random.randint(*profile['thread_range']),
    )


def generate_dataset(count=2500, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    types = list(PROCESS_PROFILES.keys())
    weights = [PROCESS_PROFILES[t]['weight'] for t in types]
    processes = []
    for pid in range(1, count + 1):
        process_type = random.choices(types, weights=weights, k=1)[0]
        arrival_time = int(np.random.exponential(scale=count * 0.1))
        processes.append(generate_process(pid, process_type, arrival_time))
    processes.sort(key=lambda p: (p.arrival_time, p.pid))
    return processes


def save_to_csv(processes, filepath='data/processes.csv'):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df = pd.DataFrame([p.to_dict() for p in processes])
    df.to_csv(filepath, index=False)
    print(f"[✓] Saved {len(processes)} processes to {filepath}")
    print(f"    Burst stats: mean={df['cpu_burst'].mean():.1f}, "
          f"min={df['cpu_burst'].min()}, max={df['cpu_burst'].max()}")


def load_from_csv(filepath='data/processes.csv'):
    df = pd.read_csv(filepath)
    processes = []
    for _, row in df.iterrows():
        p = Process(
            pid=int(row['pid']), process_type=row['process_type'],
            priority=int(row['priority']), arrival_time=int(row['arrival_time']),
            cpu_burst=int(row['cpu_burst']), io_frequency=float(row['io_frequency']),
            memory_mb=int(row['memory_mb']), num_threads=int(row['num_threads']),
        )
        processes.append(p)
    processes.sort(key=lambda p: (p.arrival_time, p.pid))
    return processes


if __name__ == '__main__':
    processes = generate_dataset(count=2500, seed=42)
    save_to_csv(processes)
