import datetime
import cv2, time
import os

# load model
model_path = './models/opencv_face_detector_uint8.pb'
config_path = './models/opencv_face_detector.pbtxt'
net = cv2.dnn.readNetFromTensorflow(model_path, config_path)

conf_threshold = 0.8    # conf_threshold 값보다 더 높이 인식된 얼굴만 추출하게 하는 파라미터값입니다.

capture = cv2.VideoCapture(0)    # 0번 카메라를 켭니다.
capture.isOpened()

fourcc = cv2.VideoWriter_fourcc(*'DIVX') 
record = False
count = 1
key = 1

global capture, count, key

def img_capture(folderName): # 이미지 저장 경로를 입력 파라미터로 받습니다.
    global capture, count
    
    if(count != 0):     # 프레임수만큼 이미지로 저장합니다. count % 15 == 0 으로 설정하면 15프레임마다 1장씩 이미지를 저장할 수 있습니다.
        try:
            cv2.imwrite(folderName+"frame%d.jpg" % (count), chk_img)
            print('Saved frame%d.jpg' % count)
        except Exception as e: # 얼굴이 잡히지 않을때 오류나는 것을 예외처리하는 함수입니다.
            print(e)
            
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
            
            # 정사각형 박스를 구하기 위한 구간
            wi = x2-x1 # 가로변의 길이
            hi = y2-y1 # 세로변의 길이
            centroidx, centroidy = x1+(wi//2), y1+(hi//2) # 얼굴박스의 중앙점
            if wi >= hi : # 가로변이 세로변 보다 길거나 같을 경우, 세로변의 길이를 가로변의 길이와 맞춥니다.
                y1 = centroidy - (wi//2)
                y2 = centroidy + (wi//2)
            else : # 세로변의 길이가 가로변보다 길 경우, 가로변의 길이를 세로변의 길이와 맞춥니다.
                x1 = centroidx - (hi//2)
                x2 = centroidx + (hi//2)
      
            # draw rects
            cv2.rectangle(result_img, (x1, y1), (x2, y2), (255, 255, 255), int(round(h/150)), cv2.LINE_AA)
            chk_img = result_img[y1:y2, x1:x2].copy()
            # cv2.putText(result_img, '%.2f%%' % (confidence * 100.), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # visualize
        cv2.imshow('result', result_img)
        key = cv2.waitKey(1)
   

        if key == 27:           # 27 = ESC
            break
            
        elif key == 24:         # 동영상 녹화 24 = Ctrl + X
            record=True
            image_path = './image/' # 이미지사진의 기본경로
            path_="./video/" # 동영상의 기본경로
            now = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S") # 이미지와 영상의 저장시작 시점을 통일하기 위해 녹화시점으로 이동
            if str(now) not in os.listdir(image_path): # 이미지파일은 시점별로 묶어야 하기때문에 해당시점의 시각으로 폴더명 생성
                os.mkdir(image_path + str(now))
                image_path = image_path + str(now)+"/" # "/"를 입력하지 않을 경우, 파일이름에 해당시점이 붙어서 파일이름만 길어지게 생성됨
            video = cv2.VideoWriter(path_+"record.mp4", fourcc, 30.0, (img.shape[1], img.shape[0]))
            
        
        elif key == 3:          # 동영상 녹화 중지 및 영상 저장 3 = Ctrl + C
            print("Record stop")
            record = False       
            video.release()

            os.rename(path_+"record.mp4",path_+str(now)+".mp4") # 정상적인 동영상 녹화 종료시 파일이름 변경
            
        if record == True:      # 동영상이 녹화되는 동안 프레임 만큼 이미지로 저장
            print("Record is running")
            video.write(result_img)
            img_capture(image_path) # 이미지 저장 경로를 입력파라미터로 전달
else:
    print("Camera is not opened")
            
capture.release()
cv2.destroyAllWindows()
