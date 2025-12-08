import time

detect = 5
alert = 3
try:
    if detect > 0:
        while detect >= 0:
            mins, secs = divmod(detect, 60)
            timer = '{:02d}:{:02d}'.format(mins, secs)
            # print(timer, end='\r')
            print(timer)
            time.sleep(1)
            detect -= 1
        print("END 1", end='\n\n')

        if detect <= 0 and alert > 0:
            while alert >= 0:
                mins, secs = divmod(alert, 60)
                timer = '{:02d}:{:02d}'.format(mins, secs)
                # print(timer, end='\r')
                print(timer)
                time.sleep(1)
                alert -= 1
            print("END 2", end='\n\n')
        else:
            print("!!! alert TIMER ERROR !!!")
    else:
        print("!!! detect TIMER ERROR !!!")
    
    print("TIMER DONE")
except Exception as e:
    print("Error:", e)

# def countdown(t):
#     while t:
#         mins, secs = divmod(t, 60)
#         timer = '{:02d}:{:02d}'.format(mins, secs)
#         print(timer)
#         time.sleep(1)
#         t -= 1
#     print("Time's up!")
# countdown(t=5)