"""
dqn_scheduler.py — DQN Reinforcement Learning Scheduler
==========================================================
Uses the trained DQN agent to make scheduling decisions.

FUNDAMENTAL DIFFERENCE:
XGBoost-SJF and LSTM-SJF predict burst times, then pick the shortest.
They are still fundamentally SJF — just with estimated bursts.

DQN doesn't predict burst times at all. Instead, it looks at the
entire ready queue state and directly picks which process to schedule.
It learned this policy through thousands of training episodes.

This means DQN could potentially discover scheduling strategies
that don't map to any named algorithm — it learns what works.
"""

import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from module1_os.process import Process
from module1_os.schedulers.base import BaseScheduler, ScheduleResult
from module2_ml.dqn_agent import DQNAgent, DQN_CONFIG


class DQNScheduler(BaseScheduler):

    def __init__(self):
        self.agent = DQNAgent()
        self.agent.load()  # Load trained model, set epsilon=0

    @property
    def name(self) -> str:
        return "DQN Scheduler"

    def _get_state(self, ready_queue: list[Process], current_time: int) -> np.ndarray:
        """Encode the current scheduling state for the DQN agent."""
        state = np.zeros(DQN_CONFIG['state_size'], dtype=np.float32)

        if not ready_queue:
            return state

        q = ready_queue
        state[0] = len(q) / DQN_CONFIG['max_queue_size']
        state[1] = np.mean([p.cpu_burst for p in q]) / 50.0
        state[2] = np.mean([p.priority for p in q]) / 20.0
        state[3] = np.mean([p.io_frequency for p in q])
        state[4] = current_time / 1000.0

        sorted_q = sorted(q, key=lambda p: p.cpu_burst)
        for i in range(min(3, len(sorted_q))):
            base = 5 + i * 3
            state[base] = sorted_q[i].cpu_burst / 50.0
            state[base + 1] = sorted_q[i].priority / 20.0
            state[base + 2] = sorted_q[i].io_frequency

        return state

    def schedule(self, processes: list[Process]) -> ScheduleResult:
        procs = self._prepare_processes(processes)
        n = len(procs)

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

            # Get state and let the DQN agent decide
            ready_procs = [p for _, p in ready_queue]
            state = self._get_state(ready_procs, current_time)
            action = self.agent.select_action(state, len(ready_queue))

            # Clamp action to valid range
            action = min(action, len(ready_queue) - 1)

            # Get the chosen process
            idx, chosen = ready_queue[action]

            # Record timing
            chosen.start_time = current_time
            chosen.response_time = chosen.start_time - chosen.arrival_time

            # Execute for actual burst
            for t in range(current_time, current_time + chosen.cpu_burst):
                execution_log.append({'time': t, 'pid': chosen.pid})

            current_time += chosen.cpu_burst
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
        )
