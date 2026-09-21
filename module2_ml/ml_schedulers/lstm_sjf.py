"""
lstm_sjf.py — LSTM-Enhanced SJF Scheduler
============================================
Uses LSTM-predicted burst times for SJF scheduling.

KEY DIFFERENCE from XGBoost-SJF:
The LSTM builds up a history of completed processes and uses that
sequence context to predict the next burst. This means its predictions
potentially improve over time as it sees more of the workload pattern.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from module1_os.process import Process
from module1_os.schedulers.base import BaseScheduler, ScheduleResult
from module2_ml.lstm_model import LSTMPredictor


class LSTMSJFScheduler(BaseScheduler):

    def __init__(self):
        self.predictor = LSTMPredictor()

    @property
    def name(self) -> str:
        return "LSTM-SJF"

    def schedule(self, processes: list[Process]) -> ScheduleResult:
        procs = self._prepare_processes(processes)
        n = len(procs)

        # Reset LSTM history for a fresh scheduling run
        self.predictor.reset_history()

        current_time = 0
        completed = 0
        executed = [False] * n
        execution_log = []

        while completed < n:
            # Build ready queue
            ready_queue = [
                (i, p) for i, p in enumerate(procs)
                if p.arrival_time <= current_time and not executed[i]
            ]

            if not ready_queue:
                execution_log.append({'time': current_time, 'pid': None})
                current_time += 1
                continue

            # Predict burst for each process in the ready queue
            # using the current LSTM history context
            predicted_bursts = {}
            for _, p in ready_queue:
                pred = self.predictor.predict(p.to_dict())
                predicted_bursts[p.pid] = pred

            # Pick process with shortest PREDICTED burst
            idx, chosen = min(ready_queue,
                              key=lambda x: predicted_bursts[x[1].pid])

            # Record timing
            chosen.start_time = current_time
            chosen.response_time = chosen.start_time - chosen.arrival_time

            # Execute for ACTUAL burst
            for t in range(current_time, current_time + chosen.cpu_burst):
                execution_log.append({'time': t, 'pid': chosen.pid})

            current_time += chosen.cpu_burst
            chosen.completion_time = current_time
            chosen.turnaround_time = chosen.completion_time - chosen.arrival_time
            chosen.waiting_time = chosen.turnaround_time - chosen.cpu_burst

            # Update LSTM history with the completed process
            # This is the key advantage — the LSTM learns from the sequence
            self.predictor.add_to_history(chosen.to_dict())

            executed[idx] = True
            completed += 1

        return ScheduleResult(
            algorithm_name=self.name,
            processes=procs,
            execution_log=execution_log,
            total_time=current_time,
        )
