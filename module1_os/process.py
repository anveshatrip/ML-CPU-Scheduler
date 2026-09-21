from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Process:
    pid: int
    process_type: str
    priority: int
    arrival_time: int
    cpu_burst: int
    io_frequency: float
    memory_mb: int
    num_threads: int

    start_time: Optional[int] = None
    completion_time: Optional[int] = None
    waiting_time: Optional[int] = None
    turnaround_time: Optional[int] = None
    response_time: Optional[int] = None
    remaining_burst: Optional[int] = None

    def reset(self):
        self.start_time = None
        self.completion_time = None
        self.waiting_time = None
        self.turnaround_time = None
        self.response_time = None
        self.remaining_burst = self.cpu_burst

    def to_dict(self):
        return {
            'pid': self.pid,
            'process_type': self.process_type,
            'priority': self.priority,
            'arrival_time': self.arrival_time,
            'cpu_burst': self.cpu_burst,
            'io_frequency': self.io_frequency,
            'memory_mb': self.memory_mb,
            'num_threads': self.num_threads,
        }

    def __repr__(self):
        return (f"Process(pid={self.pid}, type={self.process_type}, "
                f"priority={self.priority}, arrival={self.arrival_time}, "
                f"burst={self.cpu_burst})")
