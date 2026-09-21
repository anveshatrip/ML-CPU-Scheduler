from copy import deepcopy

from .base import BaseScheduler, ScheduleResult


class FCFSScheduler(BaseScheduler):

    def __init__(self):
        super().__init__("FCFS")

    def schedule(self, processes):

        # Make a copy so the original processes are not changed
        processes = deepcopy(processes)

        # Sort processes according to arrival time
        processes.sort(key=lambda p: p.arrival_time)

        current_time = 0

        for process in processes:

            # If CPU is idle, move time to process arrival
            if current_time < process.arrival_time:
                current_time = process.arrival_time

            # Process starts
            process.start_time = current_time

            # Response time
            process.response_time = (
                process.start_time - process.arrival_time
            )

            # Process executes completely
            current_time += process.cpu_burst

            # Process completes
            process.completion_time = current_time

            # Turnaround time
            process.turnaround_time = (
                process.completion_time - process.arrival_time
            )

            # Waiting time
            process.waiting_time = (
                process.turnaround_time - process.cpu_burst
            )

            # No remaining burst after completion
            process.remaining_burst = 0

        return ScheduleResult(
            processes=processes,
            total_time=current_time
        )