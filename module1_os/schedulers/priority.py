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

        while remaining:

            # Processes that have already arrived
            available = [
                p for p in remaining
                if p.arrival_time <= current_time
            ]

            # If no process is available, move time forward
            if not available:
                current_time = min(
                    p.arrival_time for p in remaining
                )
                continue

            # Select highest-priority process
            # Smaller priority number = higher priority
            process = min(
                available,
                key=lambda p: p.priority
            )

            remaining.remove(process)

            # Process starts
            process.start_time = current_time

            # Response time
            process.response_time = (
                process.start_time - process.arrival_time
            )

            # Execute completely
            current_time += process.cpu_burst

            # Completion time
            process.completion_time = current_time

            # Turnaround time
            process.turnaround_time = (
                process.completion_time - process.arrival_time
            )

            # Waiting time
            process.waiting_time = (
                process.turnaround_time - process.cpu_burst
            )

            # Process finished
            process.remaining_burst = 0

            scheduled.append(process)

        return ScheduleResult(
            processes=scheduled,
            total_time=current_time
        )