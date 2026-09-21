from copy import deepcopy
from collections import deque

from .base import BaseScheduler, ScheduleResult


class RoundRobinScheduler(BaseScheduler):

    def __init__(self, quantum=2):
        super().__init__("Round Robin")
        self.quantum = quantum

    def schedule(self, processes):

        processes = deepcopy(processes)

        # Sort by arrival time
        processes.sort(key=lambda p: p.arrival_time)

        ready_queue = deque()

        current_time = 0
        index = 0
        completed = 0

        # Keep track of the first time each process runs
        first_run = {}

        while completed < len(processes):

            # Add newly arrived processes
            while (
                index < len(processes)
                and processes[index].arrival_time <= current_time
            ):
                ready_queue.append(processes[index])
                index += 1

            # If queue is empty, move to next process arrival
            if not ready_queue:

                if index < len(processes):
                    current_time = processes[index].arrival_time
                    continue

            # Get the next process
            process = ready_queue.popleft()

            # Record first execution time
            if process.pid not in first_run:
                first_run[process.pid] = current_time
                process.start_time = current_time

                process.response_time = (
                    current_time - process.arrival_time
                )

            # Run for one quantum or until completion
            execution_time = min(
                self.quantum,
                process.remaining_burst
            )

            current_time += execution_time
            process.remaining_burst -= execution_time

            # Add newly arrived processes
            while (
                index < len(processes)
                and processes[index].arrival_time <= current_time
            ):
                ready_queue.append(processes[index])
                index += 1

            # If process has finished
            if process.remaining_burst == 0:

                process.completion_time = current_time

                process.turnaround_time = (
                    process.completion_time
                    - process.arrival_time
                )

                process.waiting_time = (
                    process.turnaround_time
                    - process.cpu_burst
                )

                completed += 1

            else:
                # Process still has work left
                ready_queue.append(process)

        return ScheduleResult(
            processes=processes,
            total_time=current_time
        )