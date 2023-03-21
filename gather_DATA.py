import cv2 as cv2
import csv
import os
import numpy as np
import mediapipe as mp
from HandModule import handTracker , cv2, pre_process_landmark



def takePic(max_iteration ):
    letter = input("new letter(0..27) : ")
    cv2.namedWindow("test")
    tracker = handTracker()
    img_counter = 0
    cam = cv2.VideoCapture(0)
    while True:
        ret, frame = cam.read()
        frame = tracker.handFinder(frame)

        landMkL= tracker.positionFinder(frame)
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("test", frame)

        k = cv2.waitKey(1)
        if k%256 == 27 or letter == 29 :
            # ESC pressed
            print("Closing ...")
            break
        elif k%256 == 32:
            # SPACE pressed
            img_counter += 1
            print(img_counter , "/{}".format(max_iteration))
            export_Landmark(landMkL , letter)
            if(img_counter == max_iteration):
                # os.chdir('Images')
                # cv2.imwrite("{}.png".format(letter) , frame)
                
                    letter = input("new letter(0..27) : ")
                    print("now letter : ", letter )
                    
                    print("now letter : ", letter )
                    img_counter = 0
     
             
    cam.release()
    cv2.destroyAllWindows()

            
    

        
def export_Landmark(landmark_list, action):
            x = pre_process_landmark(landmark_list)
            keypoints = np.array(x , dtype=object)
          
    
            keypoints.flatten()
            keypoints = np.concatenate([[action] ,   keypoints  ] )

            
            with open('cords.csv' , mode='a' , newline='' ) as f:
                csv_writer = csv.writer(f,delimiter=',' ,quotechar='"' ,quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(keypoints)

def innit_CSV():
    landMarks = ['class']
    for val in range(0,20 +1):
        landMarks += ['x{}'.format(val) , 'y{}'.format(val) ]
    with open('cords.csv' , mode='w' , newline='' ) as f:
        csv_writer = csv.writer(f,delimiter=',' ,quotechar='"' ,quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(landMarks)




takePic(50)