"""
preprocess_google_trace.py — Convert Google Cluster Trace to Our Format
=========================================================================
WHAT THIS DOES:
    Takes the raw Google Cluster Trace 2011 task_events CSV and converts it
    into our project's process format (pid, process_type, priority, arrival_time,
    cpu_burst, io_frequency, memory_mb, num_threads).

HOW IT WORKS:
    1. Load raw task_events (450K rows of scheduling lifecycle events)
    2. Match SUBMIT (event 0) → FINISH (event 4) events for each task
    3. Compute actual runtime (burst time) = finish_timestamp - schedule_timestamp
    4. Map Google's scheduling_class (0-3) to our process_type labels
    5. Map Google's normalized resource requests to our feature format
    6. Save as data/processes.csv (same format as our synthetic generator)

GOOGLE TRACE SCHEMA (task_events):
    Col 0: timestamp (microseconds since trace start)
    Col 5: event_type
        0 = SUBMIT    (task enters the system)
        1 = SCHEDULE  (task gets CPU)
        4 = FINISH    (task completes)
        2 = EVICT, 3 = FAIL, 5 = KILL (task didn't complete normally)
    Col 7: scheduling_class (0=best-effort, 1=mid, 2=production, 3=monitoring)
    Col 8: priority (0-11, higher = more important)
    Col 9: cpu_request (normalized, 0.0-1.0 of one machine's CPU)
    Col 10: memory_request (normalized, 0.0-1.0 of one machine's memory)

DATA SOURCE:
    Google Cluster Trace 2011, partition part-00000-of-00500
    https://storage.googleapis.com/clusterdata-2011-2/task_events/
    Reference: Reiss et al., "Google Cluster-Usage Traces", 2011

USAGE:
    python data/preprocess_google_trace.py
"""

import os
import pandas as pd
import numpy as np


def preprocess():
    print("=" * 60)
    print("  Preprocessing Google Cluster Trace 2011")
    print("=" * 60)

    # ─── Step 1: Load raw data ───────────────────────────────────────
    raw_path = 'data/task_events_raw.csv.gz'
    print(f"\n[1/5] Loading raw data from {raw_path}...")

    cols = ['timestamp', 'missing_info', 'job_id', 'task_index', 'machine_id',
            'event_type', 'user', 'scheduling_class', 'priority',
            'cpu_request', 'memory_request', 'disk_space_request', 'constraint']

    df = pd.read_csv(raw_path, header=None, names=cols, compression='gzip')
    print(f"      Loaded {len(df)} events")

    # ─── Step 2: Compute task runtimes ───────────────────────────────
    print("[2/5] Computing task runtimes (SCHEDULE -> FINISH)...")

    # Create a unique task ID from job_id + task_index
    df['task_id'] = df['job_id'].astype(str) + '_' + df['task_index'].astype(str)

    # Get SUBMIT events (arrival time)
    submit_events = df[df['event_type'] == 0][['task_id', 'timestamp', 'scheduling_class',
                                                 'priority', 'cpu_request', 'memory_request']].copy()
    submit_events = submit_events.drop_duplicates(subset='task_id', keep='first')
    submit_events.rename(columns={'timestamp': 'submit_time'}, inplace=True)

    # Get SCHEDULE events (when task first gets CPU)
    schedule_events = df[df['event_type'] == 1][['task_id', 'timestamp']].copy()
    schedule_events = schedule_events.drop_duplicates(subset='task_id', keep='first')
    schedule_events.rename(columns={'timestamp': 'schedule_time'}, inplace=True)

    # Get FINISH events (when task completes)
    finish_events = df[df['event_type'] == 4][['task_id', 'timestamp']].copy()
    finish_events = finish_events.drop_duplicates(subset='task_id', keep='first')
    finish_events.rename(columns={'timestamp': 'finish_time'}, inplace=True)

    # Merge: only keep tasks that were SUBMITTED, SCHEDULED, and FINISHED
    tasks = submit_events.merge(schedule_events, on='task_id', how='inner')
    tasks = tasks.merge(finish_events, on='task_id', how='inner')
    print(f"      Tasks with complete lifecycle: {len(tasks)}")

    # Compute runtime (burst time) in time units
    # Original timestamps are in microseconds — convert to reasonable time units
    # We'll normalize to a scale that makes sense for scheduling simulation
    tasks['runtime_us'] = tasks['finish_time'] - tasks['schedule_time']

    # Filter out invalid runtimes (negative or zero)
    tasks = tasks[tasks['runtime_us'] > 0].copy()
    print(f"      Tasks with positive runtime: {len(tasks)}")

    # ─── Step 3: Map to our feature format ───────────────────────────
    print("[3/5] Mapping Google features to our format...")

    # Map scheduling_class to process_type
    # Google's scheduling classes:
    # 0 = free/best-effort (like our 'interactive' — low priority background tasks)
    # 1 = best-effort/mid (like our 'io_bound' — moderate tasks)
    # 2 = production (like our 'cpu_bound' — important compute tasks)
    # 3 = monitoring/latency-sensitive (like our 'mixed' — system services)
    sclass_to_type = {
        0: 'interactive',
        1: 'io_bound',
        2: 'cpu_bound',
        3: 'mixed',
    }
    tasks['process_type'] = tasks['scheduling_class'].map(sclass_to_type)
    tasks = tasks.dropna(subset=['process_type'])

    # Normalize runtime to a 1-50 scale (matching our simulation's burst range)
    # Using percentile-based scaling to handle outliers
    p99 = tasks['runtime_us'].quantile(0.99)
    tasks['cpu_burst'] = (tasks['runtime_us'] / p99 * 49 + 1).clip(1, 50).astype(int)

    # Map priority (Google uses 0-11, we use 1-20)
    tasks['priority'] = ((tasks['priority'] / 11) * 19 + 1).clip(1, 20).astype(int)
    # Invert: Google's high priority = high number, ours = low number
    tasks['priority'] = 21 - tasks['priority']

    # CPU request → io_frequency (inverse: high CPU = low I/O)
    tasks['cpu_request'] = tasks['cpu_request'].fillna(tasks['cpu_request'].median())
    tasks['io_frequency'] = (1.0 - tasks['cpu_request'].clip(0, 1)).round(2)

    # Memory request → memory_mb (scale normalized 0-1 to 10-2048)
    tasks['memory_request'] = tasks['memory_request'].fillna(tasks['memory_request'].median())
    tasks['memory_mb'] = (tasks['memory_request'] * 2038 + 10).clip(10, 2048).astype(int)

    # Arrival time: normalize submit_time to start from 0, in reasonable time units
    min_time = tasks['submit_time'].min()
    time_range = tasks['submit_time'].max() - min_time
    tasks['arrival_time'] = ((tasks['submit_time'] - min_time) / time_range * 500).astype(int)

    # num_threads: derive from cpu_request (higher CPU request ≈ more threads)
    tasks['num_threads'] = (tasks['cpu_request'].fillna(0.025) * 30 + 1).clip(1, 16).astype(int)

    # ─── Step 4: Create final dataset ────────────────────────────────
    print("[4/5] Creating final dataset...")

    # Assign sequential PIDs
    tasks = tasks.sort_values('arrival_time').reset_index(drop=True)
    tasks['pid'] = range(1, len(tasks) + 1)

    # Select and order columns to match our Process dataclass format
    final = tasks[['pid', 'process_type', 'priority', 'arrival_time',
                    'cpu_burst', 'io_frequency', 'memory_mb', 'num_threads']].copy()

    # Take a manageable subset (2500 for training + 500 for testing = 3000)
    # Randomly sample to get a diverse mix
    if len(final) > 3000:
        final = final.sample(n=3000, random_state=42).sort_values('arrival_time').reset_index(drop=True)
        final['pid'] = range(1, len(final) + 1)

    # ─── Step 5: Save ────────────────────────────────────────────────
    print("[5/5] Saving processed data...")

    output_path = 'data/processes.csv'
    final.to_csv(output_path, index=False)

    print(f"\n[OK] Saved {len(final)} processes to {output_path}")
    print(f"\n    Dataset Statistics:")
    print(f"    -----------------------------------------")
    print(f"    Process type distribution:")
    for ptype, count in final['process_type'].value_counts().items():
        print(f"      {ptype:<15} {count:>5} ({count/len(final)*100:.1f}%)")
    print(f"    Burst time: mean={final['cpu_burst'].mean():.1f}, "
          f"min={final['cpu_burst'].min()}, max={final['cpu_burst'].max()}, "
          f"std={final['cpu_burst'].std():.1f}")
    print(f"    Priority:   mean={final['priority'].mean():.1f}, "
          f"range=[{final['priority'].min()}, {final['priority'].max()}]")
    print(f"    Memory MB:  mean={final['memory_mb'].mean():.0f}, "
          f"range=[{final['memory_mb'].min()}, {final['memory_mb'].max()}]")

    print(f"\n    Data Source: Google Cluster Trace 2011")
    print(f"    Reference:  Reiss et al., 'Google Cluster-Usage Traces'")
    print(f"    URL: https://github.com/google/cluster-data")

    return final


if __name__ == '__main__':
    preprocess()
