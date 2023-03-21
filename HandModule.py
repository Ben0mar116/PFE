
import mediapipe as mp
import cv2
import copy
import itertools

class handTracker():
    def __init__(self, mode=False, maxHands=1, detectionCon=0.5,modelComplexity=1,trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelComplex = modelComplexity
        self.trackCon = trackCon
        self.mp_drawing_styles = mp.solutions.drawing_styles
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
                        self.mpDraw.draw_landmarks(img, handLms ,self.mpHands.HAND_CONNECTIONS ,
                                                    landmark_drawing_spec=self.mpDraw.DrawingSpec(color=(0,0,0), thickness=4),
                                                    connection_drawing_spec=self.mpDraw.DrawingSpec(color=(255,255,255), thickness=4)) # identifiy hands  21
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

                
