print("Hello Atom!")

import cv2, time

# load model
detector = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

# initialize video source, default 0 (webcam)
video_path = 'videos/Practice2.mp4'
cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter('%s_output_opencv_haar.mp4' % (video_path.split('.')[0]), 
                      fourcc, 
                      cap.get(cv2.CAP_PROP_FPS),       # cv2.CAP_PROP_FPS = 프레임 속도
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

# cv2.VideoWriter(outputFile, fourcc, frame, size): 영상을 저장하기 위한 Object
# outputFile (str) – 저장될 파일명
# fourcc – Codec정보. cv2.VideoWriter_fourcc()
# frame (float) – 초당 저장될 frame
# size (list) – 저장될 사이즈(ex; 640, 480)

frame_count, tt = 1, 0

while cap.isOpened():
  ret, img = cap.read()
  if not ret:
    break

  frame_count += 1

  start_time = time.time()

  # prepare input
  result_img = img.copy()
  gray = cv2.cvtColor(result_img, cv2.COLOR_BGR2GRAY)

  # inference, find faces
  detections = detector.detectMultiScale(gray)

  # postprocessing
  for (x1, y1, w, h) in detections:
    x2 = x1 + w
    y2 = y1 + h

    # draw rects
    cv2.rectangle(result_img, (x1, y1), (x2, y2), (255, 255, 255), 2, cv2.LINE_AA)

  # inference time
  tt += time.time() - start_time
  fps = frame_count / tt
  cv2.putText(result_img, 'FPS(haar): %.2f' % (fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

  # visualize
  cv2.imshow('result', result_img)
  if cv2.waitKey(1) == ord('q'):
    break

  out.write(result_img)

cap.release()
out.release()
cv2.destroyAllWindows()






























