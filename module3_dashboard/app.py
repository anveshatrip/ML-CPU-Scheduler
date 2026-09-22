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
    Run the live benchmark on real user inputs:
    generate processes, run all 7 schedulers, compute metrics, and return results.
    """
    from module1_os.generator import generate_processes
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

    # 1. Generate workload on the fly with the exact seed and count
    processes = generate_processes(num_processes=num_processes, seed=seed)

    # 2. Instantiate all 7 schedulers
    schedulers = [
        FCFSScheduler(),
        SJFScheduler(),
        PriorityScheduler(),
        RoundRobinScheduler(quantum=quantum),
        XGBoostSJFScheduler(),
        LSTMSJFScheduler(),
        DQNScheduler(),
    ]

    all_metrics = []
    gantt_data = {}

    for scheduler in schedulers:
        result = scheduler.schedule(processes)
        m = compute_metrics(result.processes, result.total_time)

        # Standardize algorithm display name
        algo_name = scheduler.name
        if algo_name == "SJF":
            algo_name = "SJF (Known Burst)"
        elif algo_name == "Round Robin":
            algo_name = f"Round Robin (q={quantum})"

        # If execution_log is empty (Module 1 schedulers), construct from start/completion times
        log = result.execution_log
        if not log:
            log = []
            for p in sorted(result.processes, key=lambda x: x.start_time):
                for t in range(p.start_time, p.completion_time):
                    log.append({"time": t, "pid": p.pid})

        gantt_data[algo_name] = log[:80]  # First 80 ticks for clean Gantt rendering

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
        })

    # 3. Attach individual model predictions for the process table (first 20 processes)
    xgb_p, lstm_p = get_predictors()
    proc_dicts = []
    for p in processes[:20]:
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