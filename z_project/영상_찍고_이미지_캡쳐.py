import os
import datetime
import cv2, time


capture = cv2.VideoCapture(0)    # 0번 카메라를 켭니다.
capture.isOpened()

fourcc = cv2.VideoWriter_fourcc(*'DIVX') 
record = False
count = 1
key = 1

global capture, count, key

def img_capture():
    global capture, count
    if(count % 15 == 0):
        cv2.imwrite("D:/Fpjt/z_project/image/frame%d.jpg" % (count/15), frame)
        print('Saved frame%d.jpg' % count)
    count += 1
            
if capture.isOpened():
    while True:
        ret, frame = capture.read()
        
        cv2.imshow("Video", frame)
        key = cv2.waitKey(1)

        if key == 27:           # 27 = ESC
            break
            
        elif key == 24:         # 26 = Ctrl + X
            record=True
            video = cv2.VideoWriter("D:/Fpjt/z_project/image/record.mp4", fourcc, 30.0, (frame.shape[1], frame.shape[0]))
        
        elif key == 3:          # 3 = Ctrl + C
            print("Record stop")
            record = False       
            video.release()
            now = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
            path_="D:/Fpjt/z_project/image/"
            os.rename(path_+"record.mp4",path_+str(now)+".mp4")
            
        if record == True:
            print("Record is running")
            video.write(frame)
            img_capture()
else:
    print("Camera is not opened")
            
capture.release()
cv2.destroyAllWindows()
