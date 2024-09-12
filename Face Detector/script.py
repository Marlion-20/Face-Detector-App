
import cv2
from random import randrange
# load some pre-trained data on face frontal from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# To capture image from a webcam
webcam = cv2.VideoCapture(0)

# iterate over frames
while True:
    # successful read current frame
    successful_frame_read, frame= webcam.read()

    #must convert to grayscale
    grayscale_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscale_image)

    # draw rectangle on faces
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (randrange(256), randrange(256), randrange(256)), 5)

    # show the image and wait
    cv2.imshow("Marlion AI", frame)
    key = cv2.waitKey(1)

    # to close the window press Q for quit
    if key==81 or key==113:
        break

# release video capture
webcam.release()

print("code completed")
