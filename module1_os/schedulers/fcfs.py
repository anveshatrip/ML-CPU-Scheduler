from copy import deepcopy

from .base import BaseScheduler, ScheduleResult


class FCFSScheduler(BaseScheduler):

    def __init__(self, context_switch_overhead=0):
        super().__init__("FCFS", context_switch_overhead)

    def schedule(self, processes):

        processes = deepcopy(processes)
        processes.sort(key=lambda p: p.arrival_time)

        current_time = 0
        execution_log = []
        last_pid = None
        context_switches = 0

        for process in processes:

            if current_time < process.arrival_time:
                for t in range(current_time, process.arrival_time):
                    execution_log.append({"time": t, "pid": None})
                current_time = process.arrival_time
                last_pid = None

            # Context switch overhead if switching to a new process
            if last_pid is not None and last_pid != process.pid and self.context_switch_overhead > 0:
                context_switches += 1
                for _ in range(self.context_switch_overhead):
                    execution_log.append({"time": current_time, "pid": "CS"})
                    current_time += 1

            process.start_time = current_time
            process.response_time = process.start_time - process.arrival_time

            for t in range(current_time, current_time + process.cpu_burst):
                execution_log.append({"time": t, "pid": process.pid})

            current_time += process.cpu_burst
            last_pid = process.pid
            process.completion_time = current_time
            process.turnaround_time = process.completion_time - process.arrival_time
            process.waiting_time = process.turnaround_time - process.cpu_burst
            process.remaining_burst = 0

        return ScheduleResult(
            processes=processes,
            total_time=current_time,
            algorithm_name="FCFS",
            execution_log=execution_log,
            context_switches=context_switches,
        )