import time

detect = 5
alarm = 3
if detect > 0:
    while detect >= 0:
        mins, secs = divmod(detect, 60)
        timer = '{:02d}:{:02d}'.format(mins, secs)
        # print(timer, end='\r')
        print(timer)
        time.sleep(1)
        detect -= 1
    print("END 1", end='\n\n')

    if detect <= 0 and alarm > 0:
        while alarm >= 0:
            mins, secs = divmod(alarm, 60)
            timer = '{:02d}:{:02d}'.format(mins, secs)
            # print(timer, end='\r')
            print(timer)
            time.sleep(1)
            alarm -= 1
        print("END 2", end='\n\n')
    else:
        print("!!! alarm TIMER ERROR !!!")
else:
    print("!!! detect TIMER ERROR !!!")

print("TIMER DONE")

# def countdown(t):
#     while t:
#         mins, secs = divmod(t, 60)
#         timer = '{:02d}:{:02d}'.format(mins, secs)
#         print(timer)
#         time.sleep(1)
#         t -= 1
#     print("Time's up!")
# countdown(t=5)