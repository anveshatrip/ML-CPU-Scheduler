import random
import csv
import os

from .process import Process


# Generate one process
def generate_process(pid, current_time):
    process_types = ["cpu_bound", "io_bound", "mixed", "interactive"]

    process_type = random.choice(process_types)

    # Generate attributes based on process type
    if process_type == "cpu_bound":
        cpu_burst = random.randint(20, 50)
        io_frequency = random.randint(0, 2)
        memory_mb = random.randint(400, 1000)
        priority = random.randint(3, 8)

    elif process_type == "io_bound":
        cpu_burst = random.randint(1, 8)
        io_frequency = random.randint(5, 10)
        memory_mb = random.randint(100, 400)
        priority = random.randint(2, 7)

    elif process_type == "mixed":
        cpu_burst = random.randint(8, 25)
        io_frequency = random.randint(2, 5)
        memory_mb = random.randint(200, 700)
        priority = random.randint(2, 8)

    else:  # interactive
        cpu_burst = random.randint(1, 5)
        io_frequency = random.randint(4, 10)
        memory_mb = random.randint(100, 300)
        priority = random.randint(1, 4)

    num_threads = random.randint(1, 8)

    return Process(
        pid=pid,
        process_type=process_type,
        priority=priority,
        arrival_time=current_time,
        cpu_burst=cpu_burst,
        io_frequency=io_frequency,
        memory_mb=memory_mb,
        num_threads=num_threads
    )


# Generate multiple processes
def generate_processes(num_processes=2500, seed=42):
    random.seed(seed)

    processes = []
    current_time = 0

    for pid in range(1, num_processes + 1):

        # Random arrival time
        current_time += random.randint(0, 3)

        process = generate_process(pid, current_time)
        processes.append(process)

    # Sort by arrival time
    processes.sort(key=lambda p: p.arrival_time)

    return processes


# Save processes to CSV
def save_to_csv(processes, filename="data/processes.csv"):

    # Create folder if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, "w", newline="") as file:

        writer = csv.writer(file)

        # CSV header
        writer.writerow([
            "pid",
            "process_type",
            "priority",
            "arrival_time",
            "cpu_burst",
            "io_frequency",
            "memory_mb",
            "num_threads"
        ])

        # Write process data
        for p in processes:
            writer.writerow([
                p.pid,
                p.process_type,
                p.priority,
                p.arrival_time,
                p.cpu_burst,
                p.io_frequency,
                p.memory_mb,
                p.num_threads
            ])


# Load processes from CSV
def load_from_csv(filename="data/processes.csv"):

    processes = []

    with open(filename, "r") as file:

        reader = csv.DictReader(file)

        for row in reader:

            process = Process(
                pid=int(row["pid"]),
                process_type=row["process_type"],
                priority=int(row["priority"]),
                arrival_time=int(row["arrival_time"]),
                cpu_burst=int(row["cpu_burst"]),
                io_frequency=int(row["io_frequency"]),
                memory_mb=int(row["memory_mb"]),
                num_threads=int(row["num_threads"])
            )

            processes.append(process)

    return processes