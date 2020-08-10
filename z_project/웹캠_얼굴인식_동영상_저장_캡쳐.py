import datetime
import cv2, time
import os

# load model
model_path = 'D:/Fpjt/z_project/models/opencv_face_detector_uint8.pb'
config_path = 'D:/Fpjt/z_project/models/opencv_face_detector.pbtxt'
net = cv2.dnn.readNetFromTensorflow(model_path, config_path)

conf_threshold = 0.8

capture = cv2.VideoCapture(0)    # 0번 카메라를 켭니다.
capture.isOpened()

fourcc = cv2.VideoWriter_fourcc(*'DIVX') 
record = False
count = 1
key = 1

global capture, count, key

def img_capture():
    global capture, count
    if(count != 0):
        cv2.imwrite("D:/Fpjt/z_project/image/frame%d.jpg" % (count), result_img)
        print('Saved frame%d.jpg' % count)
    count += 1

            
if capture.isOpened():
    while True:
        ret, img = capture.read()
      
        # prepare input
        result_img = img.copy()
        h, w, _ = result_img.shape
        blob = cv2.dnn.blobFromImage(result_img, 1.0, (300, 300), [104, 117, 123], False, False)
        net.setInput(blob)
      
        # inference, find faces
        detections = net.forward()
      
        # postprocessing
        for i in range(detections.shape[2]):
          confidence = detections[0, 0, i, 2]
          if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
      
            # draw rects
            cv2.rectangle(result_img, (x1, y1), (x2, y2), (255, 255, 255), int(round(h/150)), cv2.LINE_AA)
            # cv2.putText(result_img, '%.2f%%' % (confidence * 100.), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # visualize
        cv2.imshow('result', result_img)
        key = cv2.waitKey(1)
   

        if key == 27:           # 27 = ESC
            break
            
        elif key == 24:         # 동영상 녹화 24 = Ctrl + X
            record=True
            video = cv2.VideoWriter("D:/Fpjt/z_project/image/record.mp4", fourcc, 30.0, (img.shape[1], img.shape[0]))
        
        elif key == 3:          # 동영상 녹화 중지 및 영상 저장 3 = Ctrl + C
            print("Record stop")
            record = False       
            video.release()
            now = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
            path_="D:/Fpjt/z_project/image/"
            os.rename(path_+"record.mp4",path_+str(now)+".mp4")
            
        if record == True:      # 동영상이 녹화되는 동안 프레임 만큼 이미지로 저장
            print("Record is running")
            video.write(result_img)
            img_capture()
else:
    print("Camera is not opened")
            
capture.release()
cv2.destroyAllWindows()
