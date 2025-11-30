import cv2 #opencv，處理影像
import mediapipe as mp #處理手部偵測，人臉偵測
import math #數學運算 (計算向量的夾角)

mp_drawing = mp.solutions.drawing_utils #mediapipe 繪圖工具。畫節點(關鍵點)和骨架(連線)
mp_drawing_styles = mp.solutions.drawing_styles #mediapipe 繪圖樣式

mp_hands = mp.solutions.hands #mediapipe 手部偵測模組

# 根據傳入的21個關鍵點(dis.txt有寫)座標，引用夾角公式計算手指角度
def hand_angle(hand_): #hand_ :finger_point的(x, y)
    angle_list = [] #儲存每根手指的角度( 夾角公式 算完的 angle_)
    # 大拇指 Thumb
    # angle_裡的減法做完後，會分別得到「v1向量的 (x[0],y[1])、v2向量的 (x[0],y[1])」，然後帶入 vector_2d_angle() 計算夾角。
    angle_ = vector_2d_angle( 
        # v1|0-2:([0|手腕關鍵點][0|x座標] - [2|_MCP拇指掌指關節][0|x座標]),([0|手腕關鍵點][0|y座標] - [2|_MCP拇指掌指關節][0|y座標])), 
        ((int(hand_[0][0]) - int(hand_[2][0])), (int(hand_[0][1]) - int(hand_[2][1]))),
        # v2|3-4:([3|_IP拇指掌指關節][0|x座標] - [4|_TIP拇指指尖][0|x座標]),([3|_IP拇指掌指關節][0|y座標] - [4|_TIP拇指指尖][0|y座標]))
        ((int(hand_[3][0]) - int(hand_[4][0])), (int(hand_[3][1]) - int(hand_[4][1])))
    )
    angle_list.append(angle_)

    # 食指 Index_Finger
    angle_ = vector_2d_angle(
        #向量1：0 > 6，(0x-6x, 0y-6y)
        ((int(hand_[0][0]) - int(hand_[6][0])),(int(hand_[0][1]) - int(hand_[6][1]))),
        #向量2：7 > 8，(7x-8x, 7y-8y)
        ((int(hand_[7][0]) - int(hand_[8][0])),(int(hand_[7][1]) - int(hand_[8][1])))
    )
    angle_list.append(angle_)

    # 中指 Middle_Finger
    angle_ = vector_2d_angle(
        #向量1：0 > 10，(0x-10x, 0y-10y)
        ((int(hand_[0][0]) - int(hand_[10][0])),(int(hand_[0][1]) - int(hand_[10][1]))),
        #向量2：11 > 12，(11x-12x, 11y-12y)
        ((int(hand_[11][0]) - int(hand_[12][0])),(int(hand_[11][1]) - int(hand_[12][1])))
    )
    angle_list.append(angle_)

    #無名指 Ring_Finger
    angle_ = vector_2d_angle(
        #向量1：0 > 14，(0x-14x, 0y-14y)
        ((int(hand_[0][0]) - int(hand_[14][0])),(int(hand_[0][1]) - int(hand_[14][1]))),
        #向量2：15 > 16，(15x-16x, 15y-16y)
        ((int(hand_[15][0]) - int(hand_[16][0])),(int(hand_[15][1]) - int(hand_[16][1])))
    )
    angle_list.append(angle_)

    #小拇指 Pinky
    angle_ = vector_2d_angle(
        #向量1：0 > 18，(0x-18x, 0y-18y)
        ((int(hand_[0][0]) - int(hand_[18][0])),(int(hand_[0][1]) - int(hand_[18][1]))),
        #向量2：19 > 20，(19x-20x, 19y-20y)
        ((int(hand_[19][0]) - int(hand_[20][0])),(int(hand_[19][1]) - int(hand_[20][1])))
    )
    angle_list.append(angle_)
    return angle_list #回傳5根手指的角度清單，順序:大拇指、食指、中指、無名指、小拇指

"""夾角公式
根據兩點座標，計算向量的夾角。**0.5:開根號。**2:平方
    兩向量內積(ax*bx + ay*by ):(v1_x * v2_x + v1_y * v2_y) 
    > 除以 兩向量長度相(1點向量:(x**2 + y**2)**5) : /(((v1_x**2 + v1_y**2)**0.5) * ((v2_x**2 + v2_y**2)**0.5))
    > 反餘弦(acos) 
    > 轉成角度(degress)
    v1=向量1，v2=向量2。[0]=x座標、[1]=y座標。"""
# 夾角公式
def vector_2d_angle(v1, v2): #從 angle_ 得到v1(x, y)、v2(x, y)
    v1_x = v1[0] #向量1(兩個 x 座標點相減)的 x 分量
    v1_y = v1[1] #向量1(兩個 y 座標點相減)的 y 分量
    v2_x = v2[0] #向量2(兩個 x 座標點相減)的 x 分量
    v2_y = v2[1] #向量2(兩個 y 座標點相減)的 y 分量
    try:
        angle_ = math.degrees(math.acos((v1_x * v2_x + v1_y * v2_y)/(((v1_x**2 + v1_y**2)**0.5) * ((v2_x**2 + v2_y**2)**0.5))))
    except:
        angle_ = 180 #當計算出錯時，將角度設為180度
    return angle_ #回傳計算出的夾角 angle_

#根據手指角度，返回對應的手勢名稱
def hand_post(finger_angle): #finger_angle: angle_list的值(5根手指的角度)
    f1 = finger_angle[0]  #大拇指
    f2 = finger_angle[1]  #食指
    f3 = finger_angle[2]  #中指
    f4 = finger_angle[3]  #無名指
    f5 = finger_angle[4]  #小拇指

    #設 angle_ < 50 表示手指「伸直」; angle_ >= 50 表示手指「彎曲」。
    if f1 < 50 and f2 >= 50 and f3 >= 50 and f4 >= 50 and f5 >= 50:
        return 'good'
    elif f1 >= 50 and f2 >= 50 and f3 < 50 and f4 >= 50 and f5 >= 50:
        return 'fxxk'
    elif f1 < 50 and f2 >= 50 and f3 < 50 and f4 < 50 and f5 < 50:
        return 'ok'
    elif f1 >= 50 and f2 >= 50 and f3 < 50 and f4 < 50 and f5 < 50:
        return 'ok'
    elif f1 >= 50 and f2 >= 50 and f3 >= 50 and f4 >= 50 and f5 >= 50:
        return '0'
    elif f1 >= 50 and f2 < 50 and f3 >= 50 and f4 >= 50 and f5 >= 50:
        return '1'
    elif f1 >= 50 and f2 < 50 and f3 < 50 and f4 >= 50 and f5 >= 50:
        return '2'
    elif f1 >= 50 and f2 < 50 and f3 < 50 and f4 < 50 and f5 > 50:
        return '3'
    elif f1 >= 50 and f2 < 50 and f3 < 50 and f4 < 50 and f5 < 50:
        return '4'
    elif f1 < 50 and f2 < 50 and f3 < 50 and f4 < 50 and f5 < 50:
        return '5'
    elif f1 < 50 and f2 >= 50 and f3 >= 50 and f4 >= 50 and f5 < 50:
        return '6'
    elif f1 < 50 and f2 < 50 and f3 >= 50 and f4 >= 50 and f5 >= 50:
        return '7'
    elif f1 < 50 and f2 < 50 and f3 < 50 and f4 >= 50 and f5 >= 50:
        return '8'
    elif f1 < 50 and f2 < 50 and f3 < 50 and f4 < 50 and f5 >= 50:
        return '9'
    else:
        return ''

cap = cv2.VideoCapture(0) #開啟攝影機，0=預設、第一台
fontFace = cv2.FONT_HERSHEY_SIMPLEX #設定字型
lineType = cv2.LINE_AA #設定字型邊框樣式

# with...as語法:自動管理資源(不用自己寫 close() 或 release())，區塊結束(或發生錯誤)後會自動釋放資源
with mp_hands.Hands( #參數設定
    model_complexity=0, #模型複雜度:0=簡單的裝置。簡單快速 -> 精準緩慢
    max_num_hands=1, #最多偵測手部數量
    min_detection_confidence=0.5, #手掌判斷信心度=模型判斷「這是否為一隻手」的信心。範圍 0~1:越高，需要圖片越清晰才會回傳結果
    min_tracking_confidence=0.5 #關鍵點追蹤信心度=偵測到手後進行關鍵點追蹤時的穩定度要求。範圍0~1
) as hands: #啟動手部偵測模組

    w, h = 800, 600 #畫面尺寸寬、高

    while True:
        ret, img = cap.read() #read()讀取每一幀影像並回傳兩參數。ret:是否成功讀取影像。img:讀取到的影像
        img = cv2.resize(img, (w, h)) #調整畫面尺寸
        
        if not ret:
            print("Cannot receive frame")
            break

        try:
            img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #將圖像(攝影機格式) BGR 轉成(2=to) RGB(mediapipe格式) 後存在 img2。
            results = hands.process(img2) # 啟動Mediapipe手部偵測模組 > 處理 img2 (分析手掌) > 偵測 21 個關鍵點 > 回傳結果
        
            if results.multi_hand_landmarks: #如果偵測到手掌。multi_hand_landmarks:21個關鍵點，a list，from 物件hands.process()。
                for hand_landmarks in results.multi_hand_landmarks: #遍歷每一隻手的關鍵點
                    finger_point = [] #儲存21個關鍵點座標(x, y)
                    fx = [] #儲存所有x座標
                    fy = [] #儲存所有y座標
                    for i in hand_landmarks.landmark: #.landmark:21個關鍵點的座標資訊，from Mediapipe。
                        # 計算21個關鍵點的座標。mediapipe 回傳的 x,y 是相對於影像寬高的比例值(0~1)，需要乘以影像寬高 w,h 取得實際座標值
                        x = i.x * w
                        y = i.y * h
                        finger_point.append((x, y)) #將每個關鍵點座標加入 finger_point 清單
                        fx.append(int(x)) #將 x 座標加入 fx 清單
                        fy.append(int(y)) #將 y 座標加入 fy 清單
                    
                    # for 完成才會進入 if。
                    if finger_point: #如果 finger_point 有值(有偵測到手指)
                        finger_angle = hand_angle(finger_point) #計算手指角度，回傳5個手指角度的清單。
                            # finger_angle 的值 = def hand_angle() 裡的 angle_list的值。
                            # finger_point → hand_angle(finger_point)（內部多次呼叫 vector_2d_angle(處理finger_point的值(x,y)））→ finger_angle

                        text = hand_post(finger_angle) #根據手指角度判斷手勢，回傳手勢名稱
                            # finger_angle → hand_post(finger_angle) → text

                        cv2.putText(img, text, (30, 120), fontFace, 5, (255, 0, 255), 10, lineType)
                            #cv2.putText(畫面, 文字, (x,y座標), 字型, 字型大小, 顏色(BGR), 粗細, 邊框樣式)

                    # 將節點和骨架繪製到影像上
                    mp_drawing.draw_landmarks(
                        img, #顯示在哪個畫面
                        hand_landmarks, #在手掌標示 21 個關鍵點
                        mp_hands.HAND_CONNECTIONS, #節點連線 > 骨架
                        mp_drawing_styles.get_default_hand_landmarks_style(), #節點樣式:大小、顏色、透明度
                        mp_drawing_styles.get_default_hand_connections_style() #骨架樣式:粗細、顏色
                    )

            cv2.imshow('Joyous img', img)#即時顯示畫面,OpenCV格式:BGR

            if cv2.waitKey(1) == ord('q'): #按下q(可以自定)鍵退出while。1:每隔1毫秒檢查一次按鍵是否被按下。
                break
        
        #出錯時印出錯誤訊息並跳出迴圈
        except Exception as e:
            print(e)
            break

cap.release() #釋放攝影機資源
cv2.destroyAllWindows() #關閉所有 OpenCV 視窗