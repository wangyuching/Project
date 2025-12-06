import time

print(f"Local time: {time.ctime()}")
start_time = time.time()
print(f"start time: {time.ctime()}")
end_time = time.time()
print(f"end time: {time.ctime()}")
print(f"Execution time: {end_time - start_time} seconds")

time_struct = time.localtime()
format_string= "%Y年%m月%d日 (%A) %H:%M:%S"
result = time.strftime(format_string, time_struct)
print(f"日期在前: {result}")