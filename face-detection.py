# import necessary packages
import cvlib as cv
import sys
import cv2
import os 

imagePath = "images/test.jpg"

# read input image
image = cv2.imread(imagePath)

# apply face detection
faces, confidences = cv.detect_face(image)

# loop through detected faces
for face,conf in zip(faces,confidences):

    (startX,startY) = face[0],face[1]
    (endX,endY) = face[2],face[3]

    # draw rectangle over face
    cv2.rectangle(image, (startX,startY), (endX,endY), (0,255,0), 2)

    label = str(format(conf * 100, '.2f')) + '%'
    cv2.putText(image, label, (startX,startY-10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255,0), 1)

# display output
# press any key to close window           
cv2.imshow("Face detection", image)
cv2.waitKey()

# release resources
cv2.destroyAllWindows()