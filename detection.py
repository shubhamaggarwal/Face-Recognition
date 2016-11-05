import numpy as np
import cv2
from matplotlib import pyplot as plt

face_cascade = cv2.CascadeClassifier('/home/shubham/opencv-2.4.13/data/haarcascades/haarcascade_frontalface_default.xml')
no_of_people=2
no_of_tests=1
maxh=0
maxw=0
img = cv2.imread("/home/shubham/Desktop/4.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.namedWindow('img',cv2.WINDOW_NORMAL)
faces = face_cascade.detectMultiScale(gray, 1.1, 5)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    crop_img = img[y:y+h, x:x+w]
    #maxh=max(maxh,h)
    #maxw=max(maxw,w)
    #cv2.imwrite(crop_img,"/media/shubham/Work/Projects/FaceRec/faces/su"+str(i)+"/"+str(j)+".jpg")
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.resizeWindow('image', 200,200)
cv2.destroyAllWindows()