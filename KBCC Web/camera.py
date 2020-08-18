import cv2
import numpy as np
model_path = './models/opencv_face_detector_uint8.pb'
config_path = './models/opencv_face_detector.pbtxt'
net = cv2.dnn.readNetFromTensorflow(model_path, config_path)
font = cv2.FONT_HERSHEY_SIMPLEX
conf_threshold = 0.8


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
    
            
    def __del__(self):
        self.video.release()


    def get_frame(self):
        
        _, fr = self.video.read()
                    
        # prepare input
        result_img = fr.copy()
        h, w, _ = result_img.shape

        blob = cv2.dnn.blobFromImage(result_img, 1.0, (300, 300), [104, 117, 123], False, False)
        
        net.setInput(blob)  
        
        # inference, find faces
        detections = net.forward()
        for i in range(detections.shape[2]):
          confidence = detections[0, 0, i, 2]
          if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)    
        
        
        try:            
            face  = result_img[x1-50:x2+50, y1-50:y2+50]
            _, jpeg = cv2.imencode('.jpg', face)
        except:
            _, jpeg = cv2.imencode('.jpg', result_img)
        finally:
            _, jpeg = cv2.imencode('.jpg', result_img)
            return jpeg.tobytes()            
