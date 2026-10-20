import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from module1_os.process import Process
from module1_os.schedulers.base import BaseScheduler, ScheduleResult
from module2_ml.xgboost_model import XGBoostPredictor


class XGBoostSJFScheduler(BaseScheduler):

    def __init__(self, context_switch_overhead=0):
        super().__init__("XGBoost-SJF", context_switch_overhead)
        self.predictor = XGBoostPredictor()

    @property
    def name(self) -> str:
        return "XGBoost-SJF"

    def schedule(self, processes: list[Process]) -> ScheduleResult:
        procs = self._prepare_processes(processes)
        n = len(procs)

        current_time = 0
        completed = 0
        executed = [False] * n
        execution_log = []
        last_pid = None
        context_switches = 0

        # Pre-predict burst times for all processes
        predicted_bursts = {}
        for p in procs:
            pred = self.predictor.predict(p.to_dict())
            predicted_bursts[p.pid] = pred

        while completed < n:
            # Build ready queue
            ready_queue = [
                (i, p) for i, p in enumerate(procs)
                if p.arrival_time <= current_time and not executed[i]
            ]

            if not ready_queue:
                execution_log.append({'time': current_time, 'pid': None})
                current_time += 1
                last_pid = None
                continue

            # Pick process with shortest PREDICTED burst (not actual!)
            idx, chosen = min(ready_queue,
                              key=lambda x: predicted_bursts[x[1].pid])

            # Context switch overhead
            if last_pid is not None and last_pid != chosen.pid and self.context_switch_overhead > 0:
                context_switches += 1
                for _ in range(self.context_switch_overhead):
                    execution_log.append({'time': current_time, 'pid': 'CS'})
                    current_time += 1

            # Record timing
            chosen.start_time = current_time
            chosen.response_time = chosen.start_time - chosen.arrival_time

            # Execute for ACTUAL burst (the real burst, not the prediction)
            for t in range(current_time, current_time + chosen.cpu_burst):
                execution_log.append({'time': t, 'pid': chosen.pid})

            current_time += chosen.cpu_burst
            last_pid = chosen.pid
            chosen.completion_time = current_time
            chosen.turnaround_time = chosen.completion_time - chosen.arrival_time
            chosen.waiting_time = chosen.turnaround_time - chosen.cpu_burst

            executed[idx] = True
            completed += 1

        return ScheduleResult(
            algorithm_name=self.name,
            processes=procs,
            execution_log=execution_log,
            total_time=current_time,
            context_switches=context_switches,
        )
