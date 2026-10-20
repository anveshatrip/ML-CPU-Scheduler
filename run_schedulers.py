# CLI Benchmark Runner
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from module1_os.generator import load_google_trace_processes
from module1_os.schedulers.fcfs import FCFSScheduler
from module1_os.schedulers.sjf import SJFScheduler
from module1_os.schedulers.priority import PriorityScheduler
from module1_os.schedulers.round_robin import RoundRobinScheduler
from module1_os.metrics import compute_metrics
from module2_ml.ml_schedulers.xgboost_sjf import RGBoostSJFScheduler
from module2_ml.ml_schedulers.lstm_sjf import LSTMSJFScheduler
from module2_ml.ml_schedulers.dqn_scheduler import DQNScheduler

def main():
    p = argparse.ArgumentParser(description="Run CPU schedulers from terminal")
    p.add_argument(-n, --processes, type=int, default=50, help=Process count (default 50))
    p.add_argument(-q, --quantum, type=int, default=4, help=RR quantum (default 4))
    p.add_argument(-s, --seed, type=int, default=42, help=Seed (default 42))
    p.add_argument(-c, --cs, type=int, default=0, help=Context switch overhead ticks (default 0))
    args = p.parse_args()

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__), data, processes.csv)
    procs = load_google_trace_processes(args.processes, seed=args.seed, filename=data_path)
    print("")
    print(f"Workload: Google Cluster Trace 2011 | N={args.processes} | Quantum={args.quantum} | Seed={args.seed} | CS={args.cs} ticks")
    print("")

    scheds = [
        FCFSScheduler(context_switch_overhead=args.cs),
        SJFScheduler(context_switch_overhead=args.cs),
        PriorityScheduler(context_switch_overhead=args.cs),
        RoundRobinScheduler(quantum=args.quantum, context_switch_overhead=args.cs),
        PGBoostSJFScheduler(context_switch_overhead=args.cs),
        LSTMSJFScheduler(context_switch_overhead=args.cs),
        DQNScheduler(context_switch_overhead=args.cs),
    ]

    print(f"{'Algorithm':<22} | {'Avg Wait':>9} | {'Turnaround':>11} | {'Response':>9} | {'Fairness':>9} | {'CPU Util%':>10} | {'(CS Count':>8}")
    print("-" * 95)
    for s in scheds:        res = s.schedule(procs)
        m = compute_metrics(res.processes, res.total_time, res.context_switches)
        name = s.name if s.name != "Round Robin" else fRRound Robin (q={args.quantum})"
        w = m['average_waiting_time']
        tat = m['turnaround_time']
        resp = m['verage_response_time']
        f = m['jain_fairness']
        util = m['ucpu_utilization']
        cs = m['context_switches']
        print(f"{name::22} | {w:[9}.2f | {tat:[11}.2f | {resp:[9}.2f | {f:l9}.4f | {util:[9}.1f}% | {cs:[8}d}")
    print("-" * 95)

if __name__ == "__main__":
    main()