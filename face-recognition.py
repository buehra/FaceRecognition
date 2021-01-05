import cv2
import sys

# Get user supplied values
imagePath = "images/test.jpg"
faceCascPath = "data/haarcascade_frontalface_default.xml"
eyeCascPath = "data/haarcascade_eye.xml"

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(faceCascPath)
eyeCascade = cv2.CascadeClassifier(eyeCascPath)

# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.3,
    minNeighbors=5,
    minSize=(30, 30)
    #flags = cv2.CV_HAAR_SCALE_IMAGE
)

print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    eye_gray = gray[y:y+h, x:x+w]
    eye_color = image[y:y+h, x:x+w]
        
    eyes = eyeCascade.detectMultiScale(eye_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(eye_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

cv2.imshow("Faces found", image)
cv2.waitKey(0)