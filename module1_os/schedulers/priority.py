from copy import deepcopy

from .base import BaseScheduler, ScheduleResult


class PriorityScheduler(BaseScheduler):

    def __init__(self):
        super().__init__("Priority")

    def schedule(self, processes):

        processes = deepcopy(processes)

        remaining = processes.copy()
        scheduled = []
        current_time = 0
        execution_log = []

        while remaining:

            available = [
                p for p in remaining
                if p.arrival_time <= current_time
            ]

            if not available:
                next_arrival = min(p.arrival_time for p in remaining)
                for t in range(current_time, next_arrival):
                    execution_log.append({"time": t, "pid": None})
                current_time = next_arrival
                continue

            # Smaller priority number = higher priority
            process = min(available, key=lambda p: p.priority)
            remaining.remove(process)

            process.start_time = current_time
            process.response_time = process.start_time - process.arrival_time

            for t in range(current_time, current_time + process.cpu_burst):
                execution_log.append({"time": t, "pid": process.pid})

            current_time += process.cpu_burst
            process.completion_time = current_time
            process.turnaround_time = process.completion_time - process.arrival_time
            process.waiting_time = process.turnaround_time - process.cpu_burst
            process.remaining_burst = 0

            scheduled.append(process)

        return ScheduleResult(
            processes=scheduled,
            total_time=current_time,
            algorithm_name="Priority",
            execution_log=execution_log,
        )