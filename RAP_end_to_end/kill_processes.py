import os
import signal

# Find all processes that include 'python' in their name
processes = os.popen('ps aux | grep python').readlines()

# Filtering out the grep process itself
processes = [p for p in processes if 'grep' not in p]

for process in processes:
    pid = int(process.split()[1])  # Extract the PID
    os.kill(pid, signal.SIGTERM)  # Send the SIGTERM signal