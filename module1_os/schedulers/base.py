from copy import deepcopy
from dataclasses import dataclass, field


@dataclass
class ScheduleResult:
    processes: list
    total_time: int
    algorithm_name: str = ""
    execution_log: list = field(default_factory=list)
    context_switches: int = 0


class BaseScheduler:

    def __init__(self, name="", context_switch_overhead=0):
        if not hasattr(self, 'name') or not self.name:
            self.name = name
        self.context_switch_overhead = max(0, int(context_switch_overhead))

    def schedule(self, processes):
        raise NotImplementedError(
            "Each scheduler must implement schedule()"
        )

    def _prepare_processes(self, processes):
        """Deep copy processes so originals are not modified."""
        return deepcopy(processes)