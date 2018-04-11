import cv2
import os
import pickle
from os import listdir
from os.path import isfile, join


def get_data():
    data=[]
    mypath = 'C:\\Users\\aadib\\Desktop\\team\\hmdb51_org\\'
    actions = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    print(actions)
    class2index={}
    for idx, action in enumerate(actions):
        class2index[action] = idx
    for action in actions:
        frame_dir = [f for f in listdir(mypath)]
        for frame in frame_dir:
            data_i=[]
            try:
                index=int(frame)
                data_inputs=[f for f in listdir(mypath+action+'\\'+frame+'\\')]
                point=[]
                for data_i in data_inputs[:-1]:
                    image=cv2.imread(mypath+action+'\\'+frame+'\\'+data)
                    image=cv2.resize(image,(32,32),interpolation=cv2.INTER_NEAREST)
                    point.append(image)
            except ValueError:
                continue
            output=np.zeros(len(class2index))
            output[class2index[action]] += 1
            data_i.append(np.array([points, output]))
        data.append(np.array(data_i))
    return data


if __name__ == '__main__':
    data=get_data()
    print("Writing")
    with open('dataset.pickle','wb') as f:
        pickle.dump(data, f)
    print("Done")
