import sys
import cv2
from using_model import YOLO_pred


class Tracker():
    def __init__(self, id, center, initial_state, initial_box, initial_confidence):
        self.id = id
        self.center = center
        self.count = 0
        self.state = initial_state
        self.box = initial_box
        self.confidence = initial_confidence

    def update_center(self, new_center):
        self.center = new_center

    def update_state(self, new_state):
        if self.confidence > .8:
            if self.state == 'down' and new_state == 'up':
                self.count += 1
    
            self.state = new_state

    def update_box(self, new_box):
        self.box = new_box

    def update_confidence(self, new_confidence):
        self.confidence = new_confidence
    
    def draw_bbox(self, image):
        x, y, w, h = self.box
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(image, (x, y + h - 30), (x + w, y + h), (255, 255, 255), -1)
        cv2.putText(image, f'id: {self.id}, {self.state}, count: {self.count}', (x, y + h - 10), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 0), 1)

    def is_the_same_as(self, box, threshold=0.4):
        x, y, w, h = box
        union = w * h + self.box[2] * self.box[3]
        leftmost, other = sorted([box, self.box], key=lambda b: b[0])
        intersection_w = leftmost[0] + leftmost[2] - other[0]
        highest, other = sorted([box, self.box], key=lambda b: b[1])
        intersection_h = highest[1] + highest[3] - other[1]
        intersection = intersection_w * intersection_h if (intersection_w > 0 and intersection_h > 0) else 0
        return intersection / union > threshold


if __name__ == '__main__':

    yolo = YOLO_pred('model.onnx', 'model_creation/data.yaml')

    if len(sys.argv) == 2 and sys.argv[1] == '-live':
        cap = cv2.VideoCapture(0)
    elif len(sys.argv) == 2:
        cap = cv2.VideoCapture(sys.argv[1]) #('videos/pullups_challenge.mp4') # video link: https://www.youtube.com/shorts/ft7VmEyvcuc?feature=share
    else:
        print('Error.')
        print('Usage 1: python3 challenge.py /path/to/video')
        print('Usage 2: python3 challenge.py -live')
        sys.exit()

    trackers = []
    n_frames = 0

    #frame_width = int(cap.get(3))
    #frame_height = int(cap.get(4))
    #size = (frame_width, frame_height) 
    #result = cv2.VideoWriter('demo_result.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size) # to create a video with the result
    
    while True:
        n_frames += 1

        ret, frame = cap.read()

        if n_frames % 3 != 0:
            continue # skipping some frames so it will be faster

        if ret == False:
            break

        boxes, confidences, classes, index = yolo.predictions_unprocessed(frame)

        if len(index) > 0:
            flattened = index.flatten()
            for ind in flattened:
                x, y, w, h = boxes[ind]
                bb_conf = confidences[ind]
                classes_id = classes[ind]
                class_name = yolo.labels[classes_id]

                id = -1
                for t in trackers:
                    if t.is_the_same_as(boxes[ind]):
                        id = t.id
                        tracker = t
                        tracker.update_confidence(bb_conf)
                        tracker.update_center((x + w/2, y + h/2))
                        tracker.update_box(boxes[ind])
                        tracker.update_state(class_name)
                
                if id == -1:
                    tracker = Tracker(len(trackers), (x + w/2, y + h/2), class_name, boxes[ind], bb_conf)
                    trackers.append(tracker)
                
                tracker.draw_bbox(frame)            
                        
        cv2.imshow('video', frame)
        #result.write(frame)
        if cv2.waitKey(1) == 27: ##esc
            break

    cv2.destroyAllWindows()
    cap.release()





