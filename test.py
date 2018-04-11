from darkflow.net.build import TFNet
import cv2
from random import randint
import numpy as np

options = {"model": "./darkflow/cfg/yolo.cfg", "load": "yolo.weights", "threshold": 0.5}

tfnet = TFNet(options)

for idx in range(1,16):
    imgcv = cv2.imread(str(idx)+".jpg")
    result = tfnet.return_predict(imgcv)
    for i in result:
        cv2.rectangle(imgcv,(i['topleft']['x'],i['topleft']['y']),(i['bottomright']['x'],i['bottomright']['y']),(randint(0,255),0,0),2)
        cv2.putText(imgcv, str(i['label']).upper(), (i['topleft']['x'],i['topleft']['y']), cv2.FONT_HERSHEY_SIMPLEX, 2, 0)
    print(idx,result)
    rv=cv2.imwrite(str(idx)+".out.jpg",imgcv)
    print(idx,rv)
#flow --model cfg/yolo-new.cfg --load bin/yolo-new.weights --demo videofile.avi --gpu 1.0
