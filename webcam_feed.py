import cv2
from application import action_predict, yolo_predict


def draw(frame,action):
    img=frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 500)
    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 2
    action='>'.join(action)
    cv2.putText(img, action.upper(),
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)
    return img


def app_start():
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False
    video=[]
    while rval:
        video.append(frame)
        if len(video) == 200:
            action_list = action_predict(video)
            video = video[1:]
            frame = draw(frame, action_list)
        frame = yolo_predict(frame)
        cv2.imshow("preview", frame)
        rval, frame = vc.read()
        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break

    cv2.destroyWindow("preview")
    vc.release()

if __name__ == '__main__':
    app_start()