from os import listdir,mkdir
from darkflow.net.build import TFNet
import cv2

options = {"model": "cfg/yolo.cfg", "load": "yolo.weights", "threshold": 0.3}

tfnet = TFNet(options)
main_path="C:\\Users\\aadib\\Desktop\\team\\hmdb51_org\\"
actions=listdir(main_path)
print("Total Number of actions:", len(actions))
for action in actions:
    print("Action:", action)
    mkdir("./data/"+action)
    folders=listdir(main_path+str(action)+"\\")
    folders_new=[]
    for i in folders:
        try:
            i_int=int(i)
            folders_new.append(i)
        except:
            pass
    folders=folders_new
    print("Total Videos:", len(folders))
    for idx,folder in enumerate(folders):
        mkdir("./data/"+action+"/"+folder)
        frames=listdir(main_path+str(action)+"\\"+str(folder)+"\\")
        frames=[(f,int(f[5:-4])) for f in frames]
        frames=sorted(frames,key=lambda x:x[1])
        frames=frames[:-1]
        frames=[f[0] for f in frames]
        print("Total Frames in video ",idx," is ",len(frames))
        for frame in frames:
            img_path=main_path+str(action)+"\\"+str(folder)+"\\"+frame
            imgcv = cv2.imread(img_path)
            result = tfnet.return_predict(imgcv)
            #print(result)
            for idx in range(0,len(result)):
                print(result)
                if(result[idx]['label']=='person'):
                    try:
                        x1, y1 = result[idx]['topleft']['x'], result[idx]['topleft']['y']
                        x2, y2 = result[idx]['bottomright']['x'], result[idx]['bottomright']['y']
                        xall,yall,channel = imgcv.shape
                        x1=min(x1-30,xall)
                        y1=min(y1-30,yall)
                        x2=min(x2+30,xall)
                        y2=min(y2+30,yall)
                        #print(xall,yall,channel)
                        #print((x1,y1),(x2,y2))
                        #cv2.rectangle(imgcv, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        imgcv_crop=imgcv[x1:x2, y1:y2]
                        imgcv_crop=cv2.resize(imgcv_crop, (xall, yall))
                        cv2.imwrite("./data/"+str(action)+"/"+str(folder)+"/"+frame,imgcv_crop)
                        #cv2.imshow("Resulting Image", imgcv)
                        #cv2.waitKey(0)
                        break
                    except Exception as e:
                        print(action,folder,frame)