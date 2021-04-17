import cv2 as cv
import numpy as np

haar_cascade =cv.CascadeClassifier('D:\machine_learning\major project\Indian_Origin_Human_Recognition\haarcascade_frontalface_default.xml')

people =['Indian','Non-Indian']
features =np.load('features.npy',allow_pickle=True)
labels =np.load('labels.npy')

face_recognizer=cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

img =cv.imread(r'D:\machine_learning\major project\Indian_Origin_Human_Recognition\pics\morgan.jpg')

gray =cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#cv.imshow('Person',gray)

#Detect the face in the image
m = -1.0
i = None
faces_rect =haar_cascade.detectMultiScale(gray,1.1,4)
for (x,y,w,h) in faces_rect:
    faces_roi=gray[y:y+h,x:x+w]

    label,confidence =face_recognizer.predict(faces_roi)
    if i is None:
        i=label
    if confidence>m:
        i=label;
        m=confidence
    

print(f'label ={people[i]} with a confidence of {m}')
cv.putText(img,str(people[i]),(20,20),cv.FONT_HERSHEY_COMPLEX,1.0,(0,255,0),2)
cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)

cv.imshow('Detected',img)
cv.waitKey(0)