''' ---------- 判斷有無吃藥動作 ---------- '''
        is_eating_medicine = False

        if left_elbow_angle < 50 and left_distence < 150 :
            is_eating_medicine = True right_distence < 150 :
            is_eating_medicine = True

'''
吃藥計數：
第一次偵測到吃藥動作，
count += 1
倒計時10s；
    在 10s 內有動作：
        count +=1
        cv2.putText("finish medcine")
        count == 0
        break to 定時
    在 10s 內無動作：
'''
        break