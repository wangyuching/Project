import time

count = 0
alarm = 0
try:
    if count == 0:
        while count <= 5:
            mins, secs = divmod(count, 60)
            timer = '{:02d}:{:02d}'.format(mins, secs)
            # print(timer, end='\r')
            print(timer)
            time.sleep(1)
            count += 1
        print("Count Stop")

        if count >= 5 and alarm == 0:
            while alarm <= 3:
                mins, secs = divmod(alarm, 60)
                timer = '{:02d}:{:02d}'.format(mins, secs)
                # print(timer, end='\r')
                print(timer)
                time.sleep(1)
                alarm += 1
            print("Flash Stop")
        else:
            print("alarm not equal to zero")
    else:
        print("count not equal to zero")
except Exception as e:
    print("Error:", e)
