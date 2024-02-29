import cv2
from using_model import YOLO_pred


yolo = YOLO_pred('secondo_tentativo.onnx', 'data.yaml')

cap = cv2.VideoCapture('predict/pullups_yt0.mp4')

prev_label = 'up'
been_down = False
counter = 0

while True:
    ret, frame = cap.read()
    if ret == False:
        break

    pred_img, label, conf = yolo.predictions(frame)
    if label == 'down' and conf > .8:
        been_down = True
    
    if label == 'up' and been_down and conf > .8:
        counter += 1
        print(counter)
        been_down = False

    cv2.rectangle(pred_img, (0, 0), (40, 40), (255, 255, 255), -1)
    cv2.putText(pred_img, str(counter), (1, 39), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
    
    cv2.imshow('video', pred_img)
    if cv2.waitKey(1) == 27: ##esc
        break

cv2.destroyAllWindows()
cap.release()
