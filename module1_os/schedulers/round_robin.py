from copy import deepcopy
from collections import deque

from .base import BaseScheduler, ScheduleResult


class RoundRobinScheduler(BaseScheduler):

    def __init__(self, quantum=2, context_switch_overhead=0):
        super().__init__("Round Robin", context_switch_overhead)
        self.quantum = quantum

    def schedule(self, processes):

        processes = deepcopy(processes)
        processes.sort(key=lambda p: p.arrival_time)

        ready_queue = deque()
        current_time = 0
        index = 0
        completed = 0
        first_run = {}
        execution_log = []  # Tick-by-tick log for Gantt / animation
        last_pid = None
        context_switches = 0

        # Enqueue all processes that arrived at time 0
        while index < len(processes) and processes[index].arrival_time <= current_time:
            ready_queue.append(processes[index])
            index += 1

        while completed < len(processes):

            # If queue is empty, jump to the next arrival
            if not ready_queue:
                if index < len(processes):
                    # Log idle ticks while CPU is waiting
                    for t in range(current_time, processes[index].arrival_time):
                        execution_log.append({"time": t, "pid": None})
                    current_time = processes[index].arrival_time
                    last_pid = None
                    while index < len(processes) and processes[index].arrival_time <= current_time:
                        ready_queue.append(processes[index])
                        index += 1
                continue

            process = ready_queue.popleft()

            # Context switch overhead if switching to a new process
            if last_pid is not None and last_pid != process.pid and self.context_switch_overhead > 0:
                context_switches += 1
                for _ in range(self.context_switch_overhead):
                    execution_log.append({"time": current_time, "pid": "CS"})
                    current_time += 1
                # Admit any processes that arrived during the context switch
                while index < len(processes) and processes[index].arrival_time <= current_time:
                    ready_queue.append(processes[index])
                    index += 1

            # Record first-run (response time)
            if process.pid not in first_run:
                first_run[process.pid] = current_time
                process.start_time = current_time
                process.response_time = current_time - process.arrival_time

            # Run for one quantum or until the process finishes
            execution_time = min(self.quantum, process.remaining_burst)

            # Log each tick of this execution slice
            for t in range(current_time, current_time + execution_time):
                execution_log.append({"time": t, "pid": process.pid})

            current_time += execution_time
            last_pid = process.pid
            process.remaining_burst -= execution_time

            if process.remaining_burst == 0:
                process.completion_time = current_time
                process.turnaround_time = process.completion_time - process.arrival_time
                process.waiting_time = process.turnaround_time - process.cpu_burst
                completed += 1

                # Admit arrivals during this quantum
                while index < len(processes) and processes[index].arrival_time <= current_time:
                    ready_queue.append(processes[index])
                    index += 1
            else:
                # Preempted — admit new arrivals first, then re-queue
                while index < len(processes) and processes[index].arrival_time <= current_time:
                    ready_queue.append(processes[index])
                    index += 1
                ready_queue.append(process)

        return ScheduleResult(
            processes=processes,
            total_time=current_time,
            algorithm_name="Round Robin",
            execution_log=execution_log,
            context_switches=context_switches,
        )