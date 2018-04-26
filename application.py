from darkflow.net.build import TFNet
from keras.models import load_model as keras_loader
import pickle
import cv2
import keras
import pandas as pd


def load_inception():
    model = keras.applications.inception_v3.InceptionV3(include_top=True, weights='imagenet', input_tensor=None,
                                                        input_shape=None, pooling=None, classes=1000)
    print(model.summary())
    return model

def load_rules():
    data = pd.read_csv('action_rules.txt',sep="|")
    rules={}
    for idx,row in data.iterrows():
        row.to_dict()
        rules[row['labels']]=row['action']
    return rules


def load_model():
    yolo_model = load_yolo()
    action_model = load_action_predictor()
    idx2actions = pickle.load(open('idx2target.pickle','rb'))
    action_rules = load_rules()
    inception = load_inception()
    return yolo_model, action_model, idx2actions, action_rules, inception


def load_yolo():
    options = {"model": "cfg/yolo.cfg", "load": "yolo.weights", "threshold": 0.3}
    tfnet = TFNet(options)
    return tfnet


def load_action_predictor():
    m = keras_loader('activity_net.h5')
    return m


def yolo_predict(tfnet, frame):
    result = tfnet.return_predict(frame)
    return result


def object_driven_action(labels , rules):
    labels = [x['label'] for x in labels]
    label_join = ','.join(labels)
    if label_join in rules.keys():
        return rules[label_join]
    else:
        return "None"


def action_predict(model, idx2actions, yolo_labels, rules, inception, video):
    labels = [x['label'] for x in yolo_labels]
    label_join = ','.join(labels)
    if label_join in rules.keys() :
        return rules[label_join]+">Object_Driven"
    else:
        video_new=[]
        for frame_i in video:
            frame_i = cv2.resize(frame_i, (299, 299))
            frame_i = frame_i.reshape(1,299,299,3)
            features = inception.predict(frame_i)
            video_new.append(features[0])
        video=video_new
        seq_len=len(video)
        video=np.array(video)
        video = video.reshape((1,seq_len,2048))
        result=model.predict(video)
        action= idx2actions[np.argmax(result)]
        return action


def yolo_draw(results, img_cp):
    for i in range(len(results)):
        x = int(results[i]['topleft']['x'])
        y = int(results[i]['topleft']['y'])
        w = int(results[i]['bottomright']['x'])
        h = int(results[i]['bottomright']['y'])
        w1 = w-x
        h1 = h-x
        if True:
            cv2.rectangle(img_cp, (x, y), (w, h), (0, 255, 0), 2)
            #cv2.rectangle(img_cp, (x - w1, y - h1 - 20), (x + w1, y - h1), (125, 125, 125), -1)
            cv2.putText(img_cp, results[i]['label'] + ' : %.2f' % results[i]['confidence'], (x + 5, y + 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return img_cp

