"""
app.py — Flask Backend Server
================================
Serves the dashboard and exposes API endpoints.

ROUTES:
    GET  /                 -> Serve the dashboard HTML page
    POST /api/run          -> Run benchmark (or return mock data)
    GET  /api/model-stats  -> Return ML model accuracy metrics

USAGE:
    python -m module3_dashboard.app
    Then open http://localhost:5000 in your browser.

MOCK MODE:
    Set USE_MOCK = True to use fake data (for UI development).
    Set USE_MOCK = False when Modules 1 and 2 are integrated.
"""

import os
from flask import Flask, render_template, jsonify, request

# ─── CONFIGURATION ──────────────────────────────────────────────────
USE_MOCK = True  # ← Set to False once Modules 1 & 2 are ready
# ────────────────────────────────────────────────────────────────────

# Flask app setup — tell Flask where templates and static files live
app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
    static_folder=os.path.join(os.path.dirname(__file__), 'static'),
)


@app.route('/')
def index():
    """Serve the main dashboard page."""
    return render_template('index.html')


@app.route('/api/run', methods=['POST'])
def run_benchmark():
    """
    Run the full benchmark: generate processes, run all schedulers, return results.

    Accepts JSON body:
        {
            "num_processes": 50,  (optional, default 50)
            "quantum": 4,         (optional, default 4)
            "seed": 42            (optional, default 42)
        }

    Returns JSON with metrics, gantt data, and process list.
    """
    if USE_MOCK:
        from module3_dashboard.mock_data import get_mock_benchmark_results
        return jsonify(get_mock_benchmark_results())

    params = request.get_json(silent=True) or {}
    num_processes = params.get('num_processes', 50)
    quantum = params.get('quantum', 4)
    seed = params.get('seed', 42)

    # ── REAL MODE (uncomment when Modules 1 & 2 are ready) ──────────
    # from module1_os.generator import generate_dataset
    # from module1_os.schedulers import (
    #     FCFSScheduler, SJFScheduler, PriorityScheduler, RoundRobinScheduler,
    # )
    # from module1_os.metrics import compute_metrics
    #
    # from module2_ml.ml_schedulers.xgboost_sjf import XGBoostSJFScheduler
    # from module2_ml.ml_schedulers.lstm_sjf import LSTMSJFScheduler
    # from module2_ml.ml_schedulers.dqn_scheduler import DQNScheduler
    #
    # # Generate processes
    # processes = generate_dataset(count=num_processes, seed=seed)
    #
    # # Run all schedulers
    # schedulers = [
    #     FCFSScheduler(),
    #     SJFScheduler(),
    #     PriorityScheduler(aging_rate=1),
    #     RoundRobinScheduler(quantum=quantum),
    #     XGBoostSJFScheduler(),
    #     LSTMSJFScheduler(),
    #     DQNScheduler(),
    # ]
    #
    # all_metrics = []
    # gantt_data = {}
    # for scheduler in schedulers:
    #     result = scheduler.schedule(processes)
    #     metrics = compute_metrics(result)
    #     all_metrics.append(metrics)
    #     gantt_data[scheduler.name] = result.execution_log
    #
    # return jsonify({
    #     "metrics": all_metrics,
    #     "gantt": gantt_data,
    #     "processes": [p.to_dict() for p in processes],
    # })
    # ─────────────────────────────────────────────────────────────────
    return jsonify({"error": "Real mode is not wired up yet. Set USE_MOCK = True."}), 501


@app.route('/api/model-stats', methods=['GET'])
def model_stats():
    """
    Return ML model performance metrics (MAE, RMSE, R², feature importances).
    """
    if USE_MOCK:
        from module3_dashboard.mock_data import get_mock_model_stats
        return jsonify(get_mock_model_stats())

    # ── REAL MODE ────────────────────────────────────────────────────
    # from module2_ml.xgboost_model import get_model_stats as xgb_stats
    # from module2_ml.lstm_model import get_model_stats as lstm_stats
    # from module2_ml.dqn_agent import get_model_stats as dqn_stats
    # return jsonify({
    #     "xgboost": xgb_stats(),
    #     "lstm": lstm_stats(),
    #     "dqn": dqn_stats(),
    # })
    # ─────────────────────────────────────────────────────────────────
    return jsonify({"error": "Real mode is not wired up yet. Set USE_MOCK = True."}), 501


# ─── ENTRY POINT ─────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 50)
    print(" AI CPU Scheduler Dashboard")
    print(f" Mode: {'MOCK DATA' if USE_MOCK else 'LIVE'}")
    print(" Open: http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, port=5000)