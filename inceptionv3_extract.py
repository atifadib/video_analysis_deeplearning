import keras
from tqdm import tqdm
from os import listdir
from keras.models import Model, Sequential
from keras.layers import LSTM, Dense, Dropout
import cv2
import numpy as np
import pickle

activity_net=Sequential()
activity_net.add(LSTM(2048,input_shape=(500,2048),return_sequences=False,dropout=0.3))
activity_net.add(Dense(1024, activation='relu'))
activity_net.add(Dropout(0.3))
activity_net.add(Dense(51, activation='relu'))
print(activity_net.summary())
activity_net.compile(loss='categorical_crossentropy', optimizer='adam')


model = keras.applications.inception_v3.InceptionV3(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
print(model.summary())

feature_extractor = Model(inputs=model.input, outputs=model.get_layer('avg_pool').output)
#imgcv = cv2.imread("C:\\Users\\aadib\\Desktop\\team\\hmdb51_org\\brush_hair\\0\\frame0.jpg")
#imgcv = cv2.resize(imgcv,(299,299))
#imgcv = imgcv.reshape(1,299,299,3)
#feature = feature_extractor.predict(imgcv)
#print("Feature len",len(feature),len(feature[0]))
#print(feature)
main_path="C:\\Users\\aadib\\Desktop\\team\\hmdb51_org\\"
actions=listdir(main_path)
print("Total Number of actions:", len(actions))
output_vector=[0]*(len(actions))
idx2actions={}
for idx,action in enumerate(actions):
    output_vector[idx] += 1
    idx2actions[idx]=action
    print("Action:", action)
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
        frames=listdir(main_path+str(action)+"\\"+str(folder)+"\\")
        frames=[(f,int(f[5:-4])) for f in frames]
        frames=sorted(frames,key=lambda x:x[1])
        frames=frames[:-1]
        frames=[f[0] for f in frames]
        print("Total Frames in video ",idx," is ",len(frames))
        video=[]
        for frame in tqdm(frames):
            img_path=main_path+str(action)+"\\"+str(folder)+"\\"+frame
            imgcv = cv2.imread(img_path)
            imgcv = cv2.resize(imgcv, (299, 299))
            imgcv = imgcv.reshape(1,299,299,3)
            feature = feature_extractor.predict(imgcv)
            feature=feature[0]
            video.append(feature)
        if(len(video)<=500):
            pad=np.zeros(2048)
            while(len(video)!=500):
                video.append(pad)
        else:
            video=video[:500]
        video=np.array(video)
        video = video.reshape((1,500,2048))
        output_vector = np.array(output_vector)
        output_vector = output_vector.reshape((1,51))
        activity_net.fit(video,output_vector,nb_epoch=20,batch_size=1,verbose=1)
        break
    break
activity_net.save('activity_net.h5')
with open('idx2actions.pickle','wb') as f:
    pickle.dump(idx2actions,f)