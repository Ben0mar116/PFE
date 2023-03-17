import cv2
import keras
import tensorflow as tf 
import copy
import itertools
import mediapipe as mp
import time
from PIL import ImageFont, ImageDraw, Image
import numpy as np
model = keras.models.load_model('saved_model/')


class handTracker():
    def __init__(self, mode=False, maxHands=1, detectionCon=0.5,modelComplexity=1,trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelComplex = modelComplexity
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.modelComplex,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils


    def handFinder(self ,img , draw = True):
        imageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #prepare image
        self.results = self.hands.process(imageRGB) # get hand landmarks
        if self.results.multi_hand_landmarks: 
                for handLms in self.results.multi_hand_landmarks:

                    if draw:
                        # draw hand landmarks
                        self.mpDraw.draw_landmarks(img, handLms ,self.mpHands.HAND_CONNECTIONS) # identifiy hands  21
        return img

    def positionFinder(self , img):
        lmlistH1 = []
      
        if self.results.multi_hand_landmarks:
            for handNO , hand_lms in enumerate(self.results.multi_hand_landmarks):
                for id, lm in enumerate(hand_lms.landmark):
                    h, w ,c= img.shape    # height width and c is channel 
                    #c will always = 3 because it represents RGB values of a pixel // the shape of an image is 3 dimentionnal tensor (height , width , c== 3 ) 
                    cx, cy = int(lm.x * w), int(lm.y * h) 
                    lmlistH1.append([cx,cy])
                    
                return lmlistH1
        return lmlistH1
                
                

        

                   



def pre_process_landmark(landmark_list):
        # Normalization

    temp_landmark_list = copy.deepcopy(landmark_list)
    

    # Convert to relative coordinates
    # take id== 0  as origin 
    base_x, base_y = 0, 0
    for id, landmark_point in enumerate(temp_landmark_list):
        if id == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
            # subtract x,y of set id to x,y of origin respectivaly
        temp_landmark_list[id][0] = temp_landmark_list[id][0] - base_x
        temp_landmark_list[id][1] = temp_landmark_list[id][1] - base_y

    # Convert to a one-dimensional list -> x , y respective to the id of hand_landmarks
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    # Normalization
    Lisrt = list(map(abs, temp_landmark_list))
    if Lisrt :
        max_value =  max(Lisrt)


        def normalize_(n): 
            return n / max_value
# to make the values in the end  between -1 and 1 
# why ? -> without preprocessing , there are some dirty data that are >> than 1  which is unfit to train a classification  model 
        temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list





                




def main():
    ALPHABET = "ابتثجحخدذرزسشصضطظعغفقكلمنهوي"
  
    cTime = 0 
    pastTime = 0
    cap = cv2.VideoCapture(0)
    image_letter = np.zeros((480, 640, 3), np.uint8)
    image_letter = cv2.cvtColor(image_letter, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_letter)
    font = ImageFont.truetype("arial.ttf", 35)
    draw = ImageDraw.Draw(pil_image)


    cv2.namedWindow('Letter', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Letter', 640, 480)
    cv2.rectangle(img=image_letter, pt1=(0, 0), pt2=(640, 480), color=(0, 0, 0), thickness=-1)
    

    tracker = handTracker()
    while cap.isOpened():
        success, image = cap.read()

        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = tracker.hands.process(imageRGB)
        # print(results.multi_hand_landmarks)
        image = tracker.handFinder(image)
        lmList = tracker.positionFinder(image)
        post_proccess = pre_process_landmark(lmList)

        if len(post_proccess) == 42 :
            predict_result = model.predict(np.array([post_proccess]))
            draw.rectangle((30, 30, 300, 150), fill=(0, 0, 0, 0))
            draw.text((30, 30), ALPHABET[int(np.argmax(np.squeeze(predict_result)))], font=font)




        # print(tracker.positionFinder2(image))
 #print  x, y position of a point /id 
        
        cTime = time.time()
        fps = 1/(cTime - pastTime)
        pastTime = cTime

        cv2.putText( image ,str(int(fps)) ,(20,30) , cv2.FONT_HERSHEY_PLAIN ,  2 ,(255 ,255 ,255) ,2 )


        cv2.imshow("Camera_Capture", image)
        image_letter = np.asarray(pil_image)
        cv2.imshow('Letter', image_letter)

        cv2.moveWindow("Camera_Capture" , 980 , 220)

        cv2.moveWindow("Letter" , 340, 220)
        

        if cv2.waitKey(1)%256 == 27:
            # ESC pressed
            print("END...")
            break
    cap.release()

    cv2.destroyAllWindows()




main()



