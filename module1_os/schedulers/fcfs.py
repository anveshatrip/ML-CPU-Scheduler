from copy import deepcopy

from .base import BaseScheduler, ScheduleResult


class FCFSScheduler(BaseScheduler):

    def __init__(self):
        super().__init__("FCFS")

    def schedule(self, processes):

        processes = deepcopy(processes)
        processes.sort(key=lambda p: p.arrival_time)

        current_time = 0
        execution_log = []

        for process in processes:

            if current_time < process.arrival_time:
                for t in range(current_time, process.arrival_time):
                    execution_log.append({"time": t, "pid": None})
                current_time = process.arrival_time

            process.start_time = current_time
            process.response_time = process.start_time - process.arrival_time

            for t in range(current_time, current_time + process.cpu_burst):
                execution_log.append({"time": t, "pid": process.pid})

            current_time += process.cpu_burst
            process.completion_time = current_time
            process.turnaround_time = process.completion_time - process.arrival_time
            process.waiting_time = process.turnaround_time - process.cpu_burst
            process.remaining_burst = 0

        return ScheduleResult(
            processes=processes,
            total_time=current_time,
            algorithm_name="FCFS",
            execution_log=execution_log,
        )