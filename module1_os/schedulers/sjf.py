from copy import deepcopy

from .base import BaseScheduler, ScheduleResult


class SJFScheduler(BaseScheduler):

    def __init__(self):
        super().__init__("SJF")

    def schedule(self, processes):

        processes = deepcopy(processes)

        # Keep track of processes that have not run yet
        remaining = processes.copy()

        scheduled = []
        current_time = 0

        while remaining:

            # Find processes that have already arrived
            available = [
                p for p in remaining
                if p.arrival_time <= current_time
            ]

            # If no process has arrived yet, move time forward
            if not available:
                current_time = min(
                    p.arrival_time for p in remaining
                )
                continue

            # Select the process with the shortest CPU burst
            process = min(
                available,
                key=lambda p: p.cpu_burst
            )

            # Remove it from the remaining list
            remaining.remove(process)

            # Process starts
            process.start_time = current_time

            # Response time
            process.response_time = (
                process.start_time - process.arrival_time
            )

            # Execute completely
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

            # Process has finished
            process.remaining_burst = 0

            scheduled.append(process)

        return ScheduleResult(
            processes=scheduled,
            total_time=current_time
        )