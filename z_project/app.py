print("Hello Atom!")

import datetime
import cv2, time

# load model
model_path = 'models/opencv_face_detector_uint8.pb'
config_path = 'models/opencv_face_detector.pbtxt'
net = cv2.dnn.readNetFromTensorflow(model_path, config_path)

conf_threshold = 0.7

# initialize video source, default 0 (webcam)
video_path = 'video/Practice2.mp4'
cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter('%s_output_opencv_dnn.mp4' % (video_path.split('.')[0]), 
                      fourcc, 
                      cap.get(cv2.CAP_PROP_FPS), 
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

frame_count, tt = 0, 0

while cap.isOpened():
  ret, img = cap.read()
  if not ret:
    break

  frame_count += 1

  start_time = time.time()

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
      cv2.putText(result_img, '%.2f%%' % (confidence * 100.), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

  # inference time
  tt += time.time() - start_time
  fps = frame_count / tt
  cv2.putText(result_img, 'FPS(dnn): %.2f' % (fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

  # visualize
  cv2.imshow('result', result_img)
  if cv2.waitKey(1) == ord('q'):
    break

  out.write(result_img)

cap.release()
out.release()
cv2.destroyAllWindows()

#############################이미지 캡쳐 과정 #################################
import os

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


















