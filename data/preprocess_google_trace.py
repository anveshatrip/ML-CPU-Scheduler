"""
preprocess_google_trace.py — Convert Google Cluster Trace to Process Format
=============================================================================
WHAT THIS DOES:
    Parses real Google Cluster Trace 2011 task event logs (gzip CSVs)
    and converts them into our project's standardized Process format:
    [pid, process_type, priority, arrival_time, cpu_burst, io_frequency, memory_mb, num_threads]

DATA SOURCE:
    Google Cluster Trace 2011 (Reiss et al., Google Inc.)
    Contains production trace logs from an ~11k machine Google Borg cluster.
    Event types:
        0 = SUBMIT    (task arrives)
        1 = SCHEDULE  (task allocated CPU)
        4 = FINISH    (task completed normally)
        5 = KILL      (task terminated/stopped)
        2 = EVICT     (task preempted/evicted)

MAPPING TO PROCESS SCHEMA:
    - scheduling_class (0-3) → process_type:
        0: 'interactive' (short, high-priority user/background requests)
        1: 'io_bound'    (moderate compute, heavy I/O waiting)
        2: 'cpu_bound'   (production compute jobs)
        3: 'mixed'       (long-running system/monitoring services)
    - cpu_burst: scaled log-runtime (finish_time - schedule_time) mapped to 1-50 range
    - priority: Google 0-11 inverted to OS priority 1-20 (1 = highest priority)
    - io_frequency: inverse of normalized CPU request [0.05, 0.95]
    - memory_mb: scaled from Google normalized memory_request [10, 2048 MB]
    - num_threads: derived from CPU request allocation [1, 16 threads]
    - arrival_time: chronologically normalized submit timestamps starting at 0
"""

import os
import glob
import numpy as np
import pandas as pd


def preprocess():
    print("=" * 60)
    print("  Processing Google Cluster Trace 2011 Dataset")
    print("=" * 60)

    # 1. Locate all available task_events parts
    data_dir = os.path.dirname(os.path.abspath(__file__))
    trace_files = sorted(glob.glob(os.path.join(data_dir, 'task_events_*.csv.gz')))
    if not trace_files:
        raise FileNotFoundError(f"No task_events_*.csv.gz files found in {data_dir}")

    print(f"\n[1/5] Loading trace archives: {[os.path.basename(f) for f in trace_files]}...")
    cols = [
        'timestamp', 'missing_info', 'job_id', 'task_index', 'machine_id',
        'event_type', 'user', 'scheduling_class', 'priority',
        'cpu_request', 'memory_request', 'disk_space_request', 'constraint'
    ]

    dfs = []
    for f in trace_files:
        df_part = pd.read_csv(f, header=None, compression='gzip', names=cols)
        dfs.append(df_part)
    all_df = pd.concat(dfs, ignore_index=True)
    print(f"      Loaded {len(all_df):,} raw cluster trace events")

    # 2. Extract complete task lifecycles
    print("\n[2/5] Extracting completed task lifecycles...")
    all_df['task_id'] = all_df['job_id'].astype(str) + '_' + all_df['task_index'].astype(str)

    # SUBMIT events (arrival & resource metadata)
    submits = all_df[all_df['event_type'] == 0][
        ['task_id', 'timestamp', 'scheduling_class', 'priority', 'cpu_request', 'memory_request']
    ].drop_duplicates('task_id', keep='first')
    submits.rename(columns={'timestamp': 'submit_time'}, inplace=True)

    # SCHEDULE events (CPU allocation timestamp)
    schedules = all_df[all_df['event_type'] == 1][['task_id', 'timestamp']].drop_duplicates('task_id', keep='first')
    schedules.rename(columns={'timestamp': 'schedule_time'}, inplace=True)

    # Completion events (4 = FINISH, 5 = KILL, 2 = EVICT)
    ends = all_df[all_df['event_type'].isin([2, 4, 5])][['task_id', 'timestamp']].sort_values('timestamp').drop_duplicates('task_id', keep='last')
    ends.rename(columns={'timestamp': 'finish_time'}, inplace=True)

    # Inner merge to keep only tasks with recorded start and end
    tasks = submits.merge(schedules, on='task_id').merge(ends, on='task_id')
    tasks['runtime_us'] = tasks['finish_time'] - tasks['schedule_time']
    tasks = tasks[tasks['runtime_us'] > 0].copy()
    print(f"      Found {len(tasks):,} tasks with verified runtime")

    # 3. Stratified balanced sampling across scheduling classes
    print("\n[3/5] Sampling balanced workload across 4 classes...")
    samples = []
    # Google classes: 0 = interactive, 1 = io_bound, 2 = cpu_bound, 3 = mixed
    targets = {0: 1000, 1: 1000, 2: 1000, 3: 1000}
    for sc, target_n in targets.items():
        grp = tasks[tasks['scheduling_class'] == sc]
        n_sample = min(len(grp), target_n)
        samples.append(grp.sample(n=n_sample, random_state=42))

    sampled = pd.concat(samples, ignore_index=True)
    # Sort chronologically by submission time to maintain real temporal ordering
    sampled = sampled.sort_values('submit_time').reset_index(drop=True)
    print(f"      Total sampled tasks: {len(sampled)}")

    # 4. Map features to OS simulation schema
    print("\n[4/5] Mapping Google attributes to scheduler process schema...")
    sclass_to_type = {
        0: 'interactive',
        1: 'io_bound',
        2: 'cpu_bound',
        3: 'mixed',
    }
    sampled['process_type'] = sampled['scheduling_class'].map(sclass_to_type)

    # Log-scaling for burst times to handle heavy-tailed datacenter distributions
    # Clamped between 1 and 50 time units for standard simulation
    log_rt = np.log10(sampled['runtime_us'].clip(lower=1000))
    p2, p98 = np.percentile(log_rt, [2, 98])
    denom = max(p98 - p2, 1e-6)
    sampled['cpu_burst'] = (1 + (log_rt.clip(p2, p98) - p2) / denom * 49).round().astype(int).clip(1, 50)

    # Priority: Google 0-11 → OS 1-20 (inverting so 1 = highest priority)
    google_prio = sampled['priority'].fillna(5).clip(0, 11)
    sampled['priority'] = (21 - ((google_prio / 11) * 19 + 1).round().astype(int)).clip(1, 20)

    # I/O frequency: inverse of CPU request (low CPU allocation = high I/O wait)
    cpu_req = sampled['cpu_request'].fillna(sampled['cpu_request'].median()).clip(0.001, 1.0)
    sampled['io_frequency'] = (1.0 - (cpu_req / cpu_req.max() * 0.9)).round(2).clip(0.05, 0.95)

    # Memory MB: normalized memory request scaled to 10 - 2048 MB
    mem_req = sampled['memory_request'].fillna(sampled['memory_request'].median()).clip(0.001, 1.0)
    sampled['memory_mb'] = (mem_req * 2038 + 10).round().astype(int).clip(10, 2048)

    # Num threads: derived from CPU request allocation
    sampled['num_threads'] = (cpu_req * 30 + 1).round().astype(int).clip(1, 16)

    # Arrival time: chronologically spaced arrivals based on submission time
    min_time = sampled['submit_time'].min()
    time_span = max(sampled['submit_time'].max() - min_time, 1)
    sampled['arrival_time'] = ((sampled['submit_time'] - min_time) / time_span * (len(sampled) * 2)).round().astype(int)

    # Sequential PIDs
    sampled['pid'] = range(1, len(sampled) + 1)

    final_cols = [
        'pid', 'process_type', 'priority', 'arrival_time',
        'cpu_burst', 'io_frequency', 'memory_mb', 'num_threads'
    ]
    final_df = sampled[final_cols].copy()

    # 5. Save output CSVs
    print("\n[5/5] Saving processed Google Cluster Trace dataset...")
    google_csv_path = os.path.join(data_dir, 'google_cluster_processes.csv')
    processes_csv_path = os.path.join(data_dir, 'processes.csv')

    final_df.to_csv(google_csv_path, index=False)
    final_df.to_csv(processes_csv_path, index=False)

    print(f"      [OK] Saved {len(final_df)} Google trace processes to:")
    print(f"           - {google_csv_path}")
    print(f"           - {processes_csv_path}")

    print("\nDataset Class Breakdown:")
    for ptype, count in final_df['process_type'].value_counts().items():
        print(f"  {ptype:<15}: {count:>5} ({count/len(final_df)*100:.1f}%)")
    print(f"Burst Time: mean={final_df['cpu_burst'].mean():.1f}, min={final_df['cpu_burst'].min()}, max={final_df['cpu_burst'].max()}")
    print(f"Priority:   mean={final_df['priority'].mean():.1f}, range=[{final_df['priority'].min()}, {final_df['priority'].max()}]")
    print(f"Memory MB:  mean={final_df['memory_mb'].mean():.0f}, range=[{final_df['memory_mb'].min()}, {final_df['memory_mb'].max()}]")

    return final_df


if __name__ == '__main__':
    preprocess()
