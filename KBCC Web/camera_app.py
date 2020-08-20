import cv2, sys, os, datetime
from PyQt5 import QtCore, QtWidgets, QtGui
from tensorflow.keras import models
import numpy as np

model_path = './models/opencv_face_detector_uint8.pb'
config_path = './models/opencv_face_detector.pbtxt'
net = cv2.dnn.readNetFromTensorflow(model_path, config_path)
model = models.load_model("./models/model_VGG19.h5")
conf_threshold = 0.5


count = 1
key = 1
nameList=['hock', 'hui', 'hye', 'ji', 'kyuong', 'moon', 'rok','sk','su','tae','yang','yoon']

class ShowVideo(QtCore.QObject):
    
    flag = 0

    camera = cv2.VideoCapture(0)

    ret, image = camera.read()
    height, width = image.shape[:2]

    VideoSignal1 = QtCore.pyqtSignal(QtGui.QImage)

    def __init__(self, parent=None):
        super(ShowVideo, self).__init__(parent)
        self.flag=0

    @QtCore.pyqtSlot()
    def startVideo(self):
        global image
        x1, x2, y1, y2 = 0, 0, 0, 0
        run_video = True
        while run_video:
            self.flag += 1
            
            ret, image = self.camera.read()
            color_swapped_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            blob = cv2.dnn.blobFromImage(color_swapped_image, 1.0, (256, 256), [104, 117, 123], False, False)
            net.setInput(blob)
            # inference, find faces
            detections = net.forward()
            
            # postprocessing
            for i in range(detections.shape[2]):
              confidence = detections[0, 0, i, 2]
              if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * self.width)
                y1 = int(detections[0, 0, i, 4] * self.height)
                x2 = int(detections[0, 0, i, 5] * self.width)
                y2 = int(detections[0, 0, i, 6] * self.height)
                
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
            if (x1 != x2 != y1 != y2):
                chk_img = color_swapped_image[y1:y2, x1:x2].copy()
            else:
                chk_img = color_swapped_image.copy()
            
            try:
                img = cv2.resize(chk_img, dsize=(256, 256), interpolation=cv2.INTER_AREA)
                img = np.expand_dims(img, axis=0)
                img = img.astype(float)/255
                pred = model.predict(img)
                who = np.argmax(pred)
            except:
                pass
            try:
                
                if np.max(pred) > conf_threshold:
                    path_ = "./static/images/today/"
                    now=str(datetime.datetime.now().strftime("%Y-%m-%d_%H_%M"))
                    cv2.rectangle(color_swapped_image, (x1, y1), (x2, y2), (255, 255, 255), int(round(self.height/150)), cv2.LINE_AA)
                    cv2.putText(color_swapped_image, '%.2f%%' % (np.max(pred) * 100.), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    print(nameList[who])
                    if now not in os.listdir(path_):
                        os.mkdir(path_+ now)
                    cv2.imwrite(path_+now+"/%s.jpg"%who,color_swapped_image)
                    if len(os.listdir(path_+now)) >= 20:
                        
                        self.flag = 0
                else:
                    print("Not Ningen",np.max(pred))
            except:
                print("Not Execute",np.max(pred), pred)
            
            qt_image1 = QtGui.QImage(color_swapped_image.data,
                                    self.width,
                                    self.height,
                                    color_swapped_image.strides[0],
                                    QtGui.QImage.Format_RGB888)
            self.VideoSignal1.emit(qt_image1)


            loop = QtCore.QEventLoop()
            QtCore.QTimer.singleShot(25, loop.quit) #25 ms
            loop.exec_()



class ImageViewer(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(ImageViewer, self).__init__(parent)
        self.image = QtGui.QImage()
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawImage(0, 0, self.image)
        self.image = QtGui.QImage()

    def initUI(self):
        self.setWindowTitle('Test')

    @QtCore.pyqtSlot(QtGui.QImage)
    def setImage(self, image):
        if image.isNull():
            print("Viewer Dropped frame!")

        self.image = image
        if image.size() != self.size():
            self.setFixedSize(image.size())
        self.update()
    
    def img_capture(folderName): # 이미지 저장 경로를 입력 파라미터로 받습니다.
        global capture, count
        
        if(count != 0):     # 프레임수만큼 이미지로 저장합니다. count % 15 == 0 으로 설정하면 15프레임마다 1장씩 이미지를 저장할 수 있습니다.
            try:
                cv2.imwrite(folderName+"frame%d.jpg" % (count), chk_img)
                print('Saved frame%d.jpg' % count)
            except Exception as e: # 얼굴이 잡히지 않을때 오류나는 것을 예외처리하는 함수입니다.
                print(e)
                
        count += 1

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)


    thread = QtCore.QThread()
    thread.start()
    vid = ShowVideo()
    vid.moveToThread(thread)

    image_viewer1 = ImageViewer()

    vid.VideoSignal1.connect(image_viewer1.setImage)
    
    push_button1 = QtWidgets.QPushButton('Start')
    push_button2 = QtWidgets.QPushButton('Stop')
    push_button1.clicked.connect(vid.startVideo)
    push_button2.clicked.connect(QtCore.QCoreApplication.instance().quit)

    vertical_layout = QtWidgets.QVBoxLayout()
    horizontal_layout = QtWidgets.QHBoxLayout()
    horizontal_layout.addWidget(image_viewer1)
    vertical_layout.addLayout(horizontal_layout)
    vertical_layout.addWidget(push_button1)
    vertical_layout.addWidget(push_button2)

    layout_widget = QtWidgets.QWidget()
    layout_widget.setLayout(vertical_layout)

    main_window = QtWidgets.QMainWindow()
    main_window.setCentralWidget(layout_widget)
    main_window.show()
    sys.exit(app.exec_())