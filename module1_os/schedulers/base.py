from copy import deepcopy
from dataclasses import dataclass, field


@dataclass
class ScheduleResult:
    processes: list
    total_time: int
    algorithm_name: str = ""
    execution_log: list = field(default_factory=list)


class BaseScheduler:

    def __init__(self, name):
        self.name = name

    def schedule(self, processes):
        raise NotImplementedError(
            "Each scheduler must implement schedule()"
        )

    def _prepare_processes(self, processes):
        """Deep copy processes so originals are not modified."""
        return deepcopy(processes)