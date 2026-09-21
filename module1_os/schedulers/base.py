from dataclasses import dataclass


@dataclass
class ScheduleResult:
    processes: list
    total_time: int


class BaseScheduler:

    def __init__(self, name):
        self.name = name

    def schedule(self, processes):
        raise NotImplementedError(
            "Each scheduler must implement schedule()"
        )