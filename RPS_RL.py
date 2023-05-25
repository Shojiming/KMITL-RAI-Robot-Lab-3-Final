import cv2
import torch
from ultralytics import YOLO
import numpy as np

model = YOLO("best.pt")

width = 640
height = 480

class_list = ['Paper', 'Rock', 'Scissors']

detection_colors = [(255,0,0),(0,255,0),(0,0,255)]

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera or file")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # resize the frame | small frame optimise the run
    frame = cv2.resize(frame, (width, height))



    # Run YOLOv8 inference on the frame
    detect_params = model.predict(source=frame,save=False,conf=0.7)

    # Convert tensor array to numpy
    DP = detect_params[0].cuda()
    DP = DP.cpu()
    DP = DP.to('cpu')
    DP = DP.numpy()

    if len(DP) != 0:
        for i in range(len(detect_params[0])):
            print(i)

            boxes = detect_params[0].boxes
            box = boxes[i]  # returns one box
            clsID = box.cls[0].cuda()
            clsID = clsID.cpu()
            clsID = clsID.to('cpu')
            clsID = clsID.numpy()
            
            conf = box.conf[0].cuda()
            conf = conf.cpu()
            conf = conf.to('cpu')
            conf = conf.numpy()
            
            bb = box.xyxy[0].cuda()
            bb = bb.cpu()
            bb = bb.to('cpu')
            bb = bb.numpy()

            cv2.rectangle(
                frame,
                (int(bb[0]), int(bb[1])),
                (int(bb[2]), int(bb[3])),
                detection_colors[int(clsID)],
                3,
            )

            # Display class name and confidence
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(
                frame,
                class_list[int(clsID)] + " " + str(np.round(conf*100, 1)) + "%",
                (int(bb[0]), int(bb[1]) - 10),
                font,
                1,
                detection_colors[int(clsID)],
                2,
            )

    # Display the resulting frame
    cv2.imshow("Rock, Paper, Scissor", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()