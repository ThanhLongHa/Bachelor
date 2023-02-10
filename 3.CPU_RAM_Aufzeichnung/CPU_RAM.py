import psutil
import time
import csv

id = 18808
process = psutil.Process(id)

filename = "../2.Bank_abwanderung/Ressourcenverbrauch/Knime_Churn2_process_usage.csv"

header = ["Timestamp", "Memory usage (MB)", "CPU usage (%)"]

with open(filename, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    while True:
        memory = process.memory_info().rss / 1024 / 1024
        cpu = process.cpu_percent() / psutil.cpu_count()

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

        writer.writerow([timestamp, memory, cpu])
        current_time = time.time()
        readable_time = time.gmtime(current_time)
        formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", readable_time)
        print("Current time:", formatted_time)
        print("Memory usage: {:.2f} MB".format(memory))
        print("CPU usage: {:.2f}%".format(cpu))
        time.sleep(1)
# def ressource_Usage(pid):
#     process = psutil.Process(pid)
#     memory = process.memory_info().rss / 1024 / 1024
#     cpu = process.cpu_percent() / psutil.cpu_count()
#     current_time = time.time()
#     readable_time = time.gmtime(current_time)
#     formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", readable_time)
#
#     print("Current time:", formatted_time)
#     print("Memory usage: {:.2f} MB".format(memory))
#     print("CPU usage: {:.2f}%".format(cpu))
#
#     time.sleep(0.5)
#
# while True:
#     ressource_Usage()
