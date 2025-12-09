import datetime
import time as t

# print(f"Local time: {dt.now()}")
# start_time = dt.now()
# print(f"start time: {start_time}")
# end_time = dt.now()
# print(f"end time: {end_time}")
# print(f"Execution time: {end_time - start_time} seconds")

# now = dt.now()
# format_string= "%Y年%m月%d日 (%A) %H:%M:%S"
# print(f"Now:  {now.strftime(format_string)}")

start = datetime.datetime.now()
endtime = start + datetime.timedelta(seconds=5)
while datetime.datetime.now() <= endtime:
    remain = (endtime - datetime.datetime.now()).seconds
    print(f"Remain :{remain} seconds")
    t.sleep(1)
