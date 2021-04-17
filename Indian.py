import os
import cv2 as cv
import numpy as np

people =['Indian', 'Non-Indian']
DIR = r'D:\machine_learning\major project\Indian_Origin_Human_Recognition\dataset_B5\dataset\train'

#for i in os.listdir(r'F:\Opencv\Photos 2'):
    #p.append(i)
#print(p)
haar_cascade =cv.CascadeClassifier('D:\machine_learning\major project\Indian_Origin_Human_Recognition\haarcascade_frontalface_default.xml')

features =[] #  Image arrays
labels =[] #Names of the people

def create_train():
    for person in people:
        path =os.path.join(DIR,person) #Finding path to a folder
        label =people.index(person) #finding the index of the person inpeople list

        for img in os.listdir(path): #looping in the person folder
            img_path =os.path.join(path,img) #finding the image path in the folder
            
            img_array =cv.imread(img_path) #reading the image of the person
            if img_array is None:
                continue
            gray =cv.cvtColor(img_array,cv.COLOR_BGR2GRAY)

            faces_rect =haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)

            for (x,y,w,h) in faces_rect:
                faces_roi =gray[y:y+h,x:x+w]
                features.append(faces_roi)
                labels.append(label)


create_train()
print('Training done -----------------')

features =np.array(features,dtype='object') #converting from lists to numpy arrays
labels =np.array(labels)

face_recognizer =cv.face.LBPHFaceRecognizer_create() #Inbuilt Face recognizer

#Train the recognizer on features list and labels list
face_recognizer.train(features,labels)

face_recognizer.save('face_trained.yml')
np.save('features.npy',features)
np.save('labels.npy',labels)