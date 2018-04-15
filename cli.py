from darkflow.net.build import TFNet
import cv2

options = {"model": "cfg/yolo.cfg", "load": "yolo.weights", "threshold": 0.9}

tfnet = TFNet(options)

imgcv = cv2.imread("./sample_img/sample_dog.jpg")
result = tfnet.return_predict(imgcv)
print(result)