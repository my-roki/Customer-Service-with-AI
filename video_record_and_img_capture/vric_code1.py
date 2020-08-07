#### 단축키 설정을 이용한 일반적인 화면캡쳐 ###

import datetime
import cv2

capture = cv2.VideoCapture("videos/Practice2.mp4")
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
record = False
# fourcc를 생성하여 디지털 미디어 포맷 코드를 생성합니다. cv2.VideoWriter_fourcc(*'코덱')을 사용하여 인코딩 방식을 설정합니다.
# record 변수를 생성하여 녹화 유/무를 설정합니다.
# Tip : FourCC(Four Character Code) : 디지털 미디어 포맷 코드입니다. 즉, 코덱의 인코딩 방식을 의미합니다.

now = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
# capture = cv2.VideoCapture(0)  
if capture.isOpened():
    while True:
        ret, frame = capture.read()
        
        cv2.imshow("Video", frame)
        key = cv2.waitKey(1)

        if key == 27:           # 27 = ESC
            break
            
        elif key == 24:         # 26 = Ctrl + X
            record=True
            video = cv2.VideoWriter("D:/Fpjt/z_project/image/"+ str(now) + ".mp4", fourcc, 30.0, (frame.shape[1], frame.shape[0]))
        elif key == 3:          # 3 = Ctrl + C
            print("Record stop")
            record = False       
            video.release()

#             path_="D:/Fpjt/z_project/image/"
#             os.rename(path_+"record.mp4",path_+str(now)+".mp4")
            
        if record == True:
            print("Record is running")
            video.write(frame)
            img_capture()
else:
    print("Camera is not opened")
            
capture.release()
cv2.destroyAllWindows()
