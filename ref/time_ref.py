import time

# print(f"Local time: {time.ctime()}")
# start_time = time.time()
# print(f"start time: {time.ctime()}")
# end_time = time.time()
# print(f"end time: {time.ctime()}")
# print(f"Execution time: {end_time - start_time} seconds")

# time_struct = time.localtime()
# format_string= "%Y年%m月%d日 (%A) %H:%M:%S"
# result = time.strftime(format_string, time_struct)
# print(f"日期在前: {result}")

# start_time = t.time()
# duration = 5
# while t.time() - start_time < duration:
#     t.sleep(1)
#     print(f"Elapsed time: {int(t.time() - start_time)} seconds")

current_time = time.time()
print(f"Time1: {(time.ctime(current_time))}")
print(f"Time2: {(time.ctime(current_time + 1))}")