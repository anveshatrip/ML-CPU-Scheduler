class Process:

    def __init__(
        self,
        pid,
        process_type,
        priority,
        arrival_time,
        cpu_burst,
        io_frequency,
        memory_mb,
        num_threads
    ):
        # Process information
        self.pid = pid
        self.process_type = process_type
        self.priority = priority
        self.arrival_time = arrival_time
        self.cpu_burst = cpu_burst
        self.io_frequency = io_frequency
        self.memory_mb = memory_mb
        self.num_threads = num_threads

        # Runtime information
        self.start_time = None
        self.completion_time = None
        self.waiting_time = 0
        self.turnaround_time = 0
        self.response_time = 0
        self.remaining_burst = cpu_burst