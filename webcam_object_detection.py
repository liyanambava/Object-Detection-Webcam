from ultralytics import YOLO
import cv2
import math 

#webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640) #width
cap.set(4, 480) #height

model = YOLO("yolo-Weights/yolov8n.pt")

classNames = model.names

while True:
    ret, img = cap.read()
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            cv2.rectangle(img, (x1, y1), (x2, y2), (44, 209, 15), 3)

            cls = int(box.cls[0])
            print("Class name:", classNames[cls])

            confidence = math.ceil((box.conf[0]*100))
            print("Confidence =",confidence,'%')
            confidence_text = f'{classNames[cls]}: {confidence}%'

            cv2.putText(img, confidence_text, [x1, y1], cv2.FONT_HERSHEY_TRIPLEX, 1, (23, 30, 227), 2) #img, text, org, font, fontscale, color, thickness

    cv2.imshow('Webcam', img)
    c = cv2.waitKey(1)
    if c == 27: #Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()