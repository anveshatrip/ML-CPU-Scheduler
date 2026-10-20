"""
app.py — Flask Backend Server (Live Mode)
=========================================
Serves the dashboard and executes the scheduling pipeline dynamically.
Connects Module 1 (OS simulation), Module 2 (ML models), and Module 3 (Dashboard).
"""

import os
import json
from flask import Flask, render_template, jsonify, request

# Flask app setup
app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
    static_folder=os.path.join(os.path.dirname(__file__), 'static'),
)

# ─── PRE-LOAD PREDICTORS FOR SPEED ──────────────────────────────────
_xgb_predictor = None
_lstm_predictor = None

def get_predictors():
    global _xgb_predictor, _lstm_predictor
    if _xgb_predictor is None:
        try:
            from module2_ml.xgboost_model import XGBoostPredictor
            _xgb_predictor = XGBoostPredictor()
        except Exception as e:
            print("Warning: Could not load XGBoostPredictor:", e)
    if _lstm_predictor is None:
        try:
            from module2_ml.lstm_model import LSTMPredictor
            _lstm_predictor = LSTMPredictor()
        except Exception as e:
            print("Warning: Could not load LSTMPredictor:", e)
    return _xgb_predictor, _lstm_predictor


@app.route('/')
def index():
    """Serve the main dashboard page."""
    return render_template('index.html')


@app.route('/api/run', methods=['POST'])
def run_benchmark():
    """
    Run the live benchmark using real Google Cluster Trace 2011 data.
    Loads processes from the preprocessed trace, runs all 7 schedulers,
    computes metrics, and returns results.
    """
    from module1_os.generator import load_google_trace_processes
    from module1_os.schedulers.fcfs import FCFSScheduler
    from module1_os.schedulers.sjf import SJFScheduler
    from module1_os.schedulers.priority import PriorityScheduler
    from module1_os.schedulers.round_robin import RoundRobinScheduler
    from module1_os.metrics import compute_metrics
    from module2_ml.ml_schedulers.xgboost_sjf import XGBoostSJFScheduler
    from module2_ml.ml_schedulers.lstm_sjf import LSTMSJFScheduler
    from module2_ml.ml_schedulers.dqn_scheduler import DQNScheduler

    params = request.get_json(silent=True) or {}
    num_processes = int(params.get('num_processes') or 50)
    quantum = int(params.get('quantum') or 4)
    seed = int(params.get('seed') or 42)
    context_switch = int(params.get('context_switch') or 0)

    # 1. Load real processes from Google Cluster Trace 2011 dataset
    processes = load_google_trace_processes(
        num_processes=num_processes, seed=seed,
        filename=os.path.join(os.path.dirname(__file__), '..', 'data', 'processes.csv')
    )

    # 2. Instantiate all 7 schedulers with context switch overhead
    schedulers = [
        FCFSScheduler(context_switch_overhead=context_switch),
        SJFScheduler(context_switch_overhead=context_switch),
        PriorityScheduler(context_switch_overhead=context_switch),
        RoundRobinScheduler(quantum=quantum, context_switch_overhead=context_switch),
        XGBoostSJFScheduler(context_switch_overhead=context_switch),
        LSTMSJFScheduler(context_switch_overhead=context_switch),
        DQNScheduler(context_switch_overhead=context_switch),
    ]

    all_metrics = []
    gantt_data = {}

    for scheduler in schedulers:
        result = scheduler.schedule(processes)
        m = compute_metrics(result.processes, result.total_time, result.context_switches)

        # Standardize algorithm display name
        algo_name = scheduler.name
        if algo_name == "SJF":
            algo_name = "SJF (Known Burst)"
        elif algo_name == "Round Robin":
            algo_name = f"Round Robin (q={quantum})"

        # Use the scheduler's execution_log for the Gantt (tick-accurate, shows preemptions and CS)
        gantt_data[algo_name] = result.execution_log[:80]

        all_metrics.append({
            "algorithm": algo_name,
            "avg_waiting_time": round(m["average_waiting_time"], 2),
            "avg_turnaround_time": round(m["average_turnaround_time"], 2),
            "avg_response_time": round(m["average_response_time"], 2),
            "cpu_utilization": round(m["cpu_utilization"], 1),
            "throughput": round(m["throughput"], 4),
            "fairness_index": round(m["jain_fairness"], 4),
            "max_waiting_time": m["max_waiting_time"],
            "total_time": result.total_time,
            "context_switches": result.context_switches,
        })

    # 3. Attach individual model predictions for all processes
    xgb_p, lstm_p = get_predictors()
    proc_dicts = []
    for p in processes:
        d = p.to_dict()
        d['predicted_burst_xgb'] = round(float(xgb_p.predict(d)), 1) if xgb_p else d['cpu_burst']
        d['predicted_burst_lstm'] = round(float(lstm_p.predict(d)), 1) if lstm_p else d['cpu_burst']
        proc_dicts.append(d)

    return jsonify({
        "metrics": all_metrics,
        "gantt": gantt_data,
        "processes": proc_dicts,
    })


@app.route('/api/model-stats', methods=['GET'])
def model_stats():
    """
    Return real ML model performance metrics from trained saved stats.
    """
    stats = {}
    saved_dir = os.path.join(os.path.dirname(__file__), '..', 'module2_ml', 'saved_models')
    for name in ['xgboost', 'lstm', 'dqn']:
        path = os.path.join(saved_dir, f'{name}_stats.json')
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    stats[name] = json.load(f)
            except Exception as e:
                print(f"Error reading {name} stats:", e)

    # If stats not found, fallback to mock stats
    if not stats:
        from module3_dashboard.mock_data import get_mock_model_stats
        return jsonify(get_mock_model_stats())

    return jsonify(stats)


# ─── ENTRY POINT ─────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 50)
    print(" AI CPU Scheduler Dashboard (LIVE MODE)")
    print(" Open: http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, port=5000)