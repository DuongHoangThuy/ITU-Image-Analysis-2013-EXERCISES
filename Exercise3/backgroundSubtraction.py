from pylab import *
import cv2
import Test1 as t

#load the video
cap = cv2.VideoCapture("/Developer/GitRepos/ITU-Image-Analysis-2013-EXERCISES/Exercise3/GreenScreen(lowMP4).mp4")
cv2.namedWindow("input")

#Get the first frame and get the video size
f, backGround = cap.read()
(w,h,t)= shape(backGround)

#Load an image as the stationary backGround
backGround=cv2.imread("/Developer/GitRepos/ITU-Image-Analysis-2013-EXERCISES/Exercise3/GreenScreen(lowMP4)_Green.jpg", cv2.COLOR_RGB2GRAY)
backGround=cv2.resize(backGround, (h,w))
[RBack, GBack, BBack] = cv2.split(backGround)
backGround=cv2.cvtColor(backGround, cv2.COLOR_RGB2GRAY)


newBackGround = cv2.imread("GreenScreen(lowMP4)_Background.jpg")
newBackGround = cv2.resize(newBackGround, (h,w))


while(f):
    f, img = cap.read()
    
    if f==True:
        img_backup = copy(img)
        [R,G,B] = cv2.split(img)
        
        R = cv2.absdiff(R, RBack) #Subtracting each color channel of current frame with each channel of bcg 
        G = cv2.absdiff(G, GBack)
        B = cv2.absdiff(B, BBack)
        
        img = cv2.addWeighted(R, 0.5, G, 0.5,0) #Merging channels
        img = cv2.addWeighted(img, 0.5, B, 0.5,0) 
        
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        #img = cv2.absdiff(img, backGround)
        (thresh, thr) = cv2.threshold(img, 40, 255, cv2.THRESH_BINARY) #Applying threshold
        
        
        
        msk = 255 - thr
    
        print msk 
        
        msk = cv2.cvtColor(msk, cv2.COLOR_GRAY2RGB)
        thr = cv2.cvtColor(thr, cv2.COLOR_GRAY2RGB)
        
        I1 = cv2.bitwise_and(img_backup, thr)
        I2 = cv2.bitwise_and(newBackGround, msk)
        
        Imixed = cv2.bitwise_or(I1,I2)
        
        cv2.imshow("input", msk)
        ch = cv2.waitKey(1)
        
        if ch == 32:#  Break by SPACE key
            break
#--------------------


