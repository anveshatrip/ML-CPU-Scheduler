def compute_metrics(processes, total_time):
    """
    Calculate performance metrics for a scheduling algorithm.
    """

    if not processes:
        return {
            "average_waiting_time": 0,
            "average_turnaround_time": 0,
            "average_response_time": 0,
            "cpu_utilization": 0,
            "throughput": 0,
            "jain_fairness": 0,
            "max_waiting_time": 0
        }

    # Average Waiting Time
    avg_waiting = sum(
        p.waiting_time for p in processes
    ) / len(processes)

    # Average Turnaround Time
    avg_turnaround = sum(
        p.turnaround_time for p in processes
    ) / len(processes)

    # Average Response Time
    avg_response = sum(
        p.response_time for p in processes
    ) / len(processes)

    # Total CPU burst time
    total_cpu_time = sum(
        p.cpu_burst for p in processes
    )

    # CPU Utilization
    if total_time > 0:
        cpu_utilization = (total_cpu_time / total_time) * 100
    else:
        cpu_utilization = 0

    # Throughput
    if total_time > 0:
        throughput = len(processes) / total_time
    else:
        throughput = 0

    # Maximum waiting time
    max_waiting = max(
        p.waiting_time for p in processes
    )

    # Jain's Fairness Index
    waiting_times = [
        p.waiting_time for p in processes
    ]

    sum_waiting = sum(waiting_times)
    sum_squared = sum(
        waiting_time ** 2
        for waiting_time in waiting_times
    )

    if sum_squared > 0:
        jain_fairness = (
            sum_waiting ** 2
        ) / (
            len(waiting_times) * sum_squared
        )
    else:
        jain_fairness = 1.0

    return {
        "average_waiting_time": avg_waiting,
        "average_turnaround_time": avg_turnaround,
        "average_response_time": avg_response,
        "cpu_utilization": cpu_utilization,
        "throughput": throughput,
        "jain_fairness": jain_fairness,
        "max_waiting_time": max_waiting
    }