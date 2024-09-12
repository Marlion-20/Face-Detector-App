import cv2
from random import randrange
# load some pre-trained data on face frontal from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# choose an image to detect faces in
image = cv2.imread("mult.PNG")

# must convert to grayscale
grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscale_image)
print(face_coordinates)

# draw rectangle on faces
for (x,y,w,h) in face_coordinates:
    cv2.rectangle(image,(x,y),(x+w, y+h), (randrange(256),randrange(256),randrange(256)),2)

# show the image and wait
cv2.imshow("Marlion AI",image)
cv2.waitKey()

print("code completed")
