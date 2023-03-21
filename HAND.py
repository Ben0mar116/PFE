from HandModule import handTracker , cv2, pre_process_landmark
import keras
import tensorflow as tf 
import textwrap
import time
from PIL import ImageFont, ImageDraw, Image
import numpy as np


model = keras.models.load_model('saved_model/')


def font_ar(number = 65):
    return ImageFont.truetype("Uthmaniac.otf", number)



def print_time( threadName, delay , model):
   time.sleep(delay)
      


def main():
    ALPHABET = "ابتثجحخدذرزسشصضطظعغفقكلمنهوي"
  
    cTime = 0 
    pastTime = 0
    cap = cv2.VideoCapture(0)
    tracker = handTracker()

    image_letter = np.zeros((640, 480, 3), np.uint8)
    image_letter = cv2.cvtColor(image_letter, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_letter)
    draw = ImageDraw.Draw(pil_image)


    cv2.namedWindow('Letter', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Letter', 640, 480)
    cv2.rectangle(img=image_letter, pt1=(0, 0), pt2=(640, 480), color=(0, 0, 0), thickness=-1)
    text = ""
    text_color = "#FFFFFF"
    mode_color = (0,255,255) 

    while cap.isOpened():
        Write = False
        # cal FPS
        cTime = time.time()
        fps = 1/(cTime - pastTime)
        pastTime = cTime

        k = cv2.waitKey(1)
        success, image = cap.read()
        
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = tracker.hands.process(imageRGB)
        # print(results.multi_hand_landmarks)
        image = tracker.handFinder(image)
        lmList = tracker.positionFinder(image)
        post_proccess = pre_process_landmark(lmList)

        draw.text((30, 450), "[space] : write.",  font = ImageFont.FreeTypeFont('arial.ttf' , 30) , fill = mode_color, stroke_width=0 )

        draw.text((260, 450), "[r] : reset.",  font = ImageFont.FreeTypeFont('arial.ttf' , 30) , fill = mode_color, stroke_width=0 )
        
        if k%256  == 114:
            # r pressed  
                mode_color = (0,0,0) # mode disapears
                draw.rectangle((0, 0, 640, 600), fill=(0, 0, 0, 0))
                Write = False
                text =""
        if len(post_proccess) == 42 :
            predict_result = model.predict(np.array([post_proccess]))
            draw.rectangle((0, 0, 300, 300), fill=(0, 0, 0, 0))
            draw.text((30, 30),   ALPHABET[int(np.argmax(np.squeeze(predict_result)))], font=font_ar() , fill = text_color)


            if k%256  == 9 and text.strip() != "":
                # TAB pressed  
                draw.rectangle((0, 0, 640, 600), fill=(0, 0, 0, 0))
                text+=" "
                draw.text ( (30,350), text, font=font_ar(65), fill=(255,255,255),spacing=5,direction='rtl',align='left',features='rtla')
            

                
            
            if k%256  == 8:
            # BACKSPACE pressed  
                draw.rectangle((0, 0, 640, 600), fill=(0, 0, 0, 0))
                text = text[0:len(text)-1]
                draw.text ( (30,350), text, font=font_ar(65), fill=(255,255,255),spacing=5,direction='rtl',align='left',features='rtla')
            
            if k%256  == 32:
            # SPACE pressed 
                mode_color = (0,0,0) # mode disapears
                Write = True

            if Write :
                # take moment shot of the alphabet value then compare it 3 sec later // if it fits -> write down
                draw.rectangle((0, 0, 600, 600), fill=(0, 0, 0, 0))
                xt1 =  int(np.argmax(np.squeeze(predict_result)))
                #  now capture letter in 3 seconds
                text +=ALPHABET[xt1]
                draw.text ( (30,350), text, font=font_ar(65), fill=(255,255,255),spacing=5,direction='rtl',align='left',features='rtla')
        
       

        cv2.putText( image ,str(int(fps)) ,(20,30) , cv2.FONT_HERSHEY_PLAIN ,  2 ,(0 ,0 ,0) ,2 )


        cv2.imshow("Camera_Capture", image)
        image_letter = np.asarray(pil_image)
        cv2.imshow('Letter', image_letter)

        cv2.moveWindow("Camera_Capture" , 980 , 220)

        cv2.moveWindow("Letter" , 340, 220)
        

        if k%256 == 27:
            # ESC pressed
            print("END...")
            break
    cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()



