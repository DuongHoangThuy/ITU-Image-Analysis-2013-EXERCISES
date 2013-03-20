import cv2
import cv
import pylab
from SIGBTools import RegionProps
from SIGBTools import getLineCoordinates
from SIGBTools import ROISelector
from SIGBTools import getImageSequence
from SIGBTools import getCircleSamples
import numpy as np
import sys
import math

from scipy.cluster.vq import *
from scipy.misc import *
from matplotlib.pyplot import *
from numpy.core.numeric import ndarray




inputFile = "/Developer/GitRepos/ITU-Image-Analysis-2013-EXERCISES/MandatoryAssignment1/Sequences/eye1.avi"
outputFile = "eyeTrackerResult.mp4"

#--------------------------
#         Global variable
#--------------------------
global imgOrig,leftTemplate,rightTemplate,frameNr
imgOrig = []
#These are used for template matching
leftTemplate = []
rightTemplate = []
frameNr =0


def GetPupil(gray,thr):
    tempResultImg = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR) #used to draw temporary results

    props = RegionProps()
    val,binI = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY_INV)
    
    #Combining Closing and Opening to the thresholded image
    st7 = cv2.getStructuringElement(cv2.MORPH_CROSS,(7,7))
    st9 = cv2.getStructuringElement(cv2.MORPH_CROSS,(9,9))
    st15 = cv2.getStructuringElement(cv2.MORPH_CROSS,(15,15))
             
    binI = cv2.morphologyEx(binI, cv2.MORPH_CLOSE, st9) #Close 
    binI= cv2.morphologyEx(binI, cv2.MORPH_OPEN, st15) #Open
    binI = cv2.morphologyEx(binI, cv2.MORPH_DILATE, st7, iterations=2) #Dialite  
    
    cv2.imshow("ThresholdPupil",binI)
    #Calculate blobs
    sliderVals = getSliderVals() #Getting slider values
    contours, hierarchy = cv2.findContours(binI, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) #Finding contours/candidates for pupil blob
    pupils = []
    pupilEllipses = []
    for cnt in contours:
        values = props.CalcContourProperties(cnt,['Area','Length','Centroid','Extend','ConvexHull']) #BUG - Add cnt.astype('int') in Windows
        if values['Area'] < sliderVals['maxSizePupil'] and values['Area'] > sliderVals['minSizePupil'] and values['Extend'] < 0.9:
            pupils.append(values)
            centroid = (int(values['Centroid'][0]),int(values['Centroid'][1]))
            cv2.circle(tempResultImg,centroid, 2, (0,0,255),4)
            pupilEllipses.append(cv2.fitEllipse(cnt))
    cv2.imshow("TempResults",tempResultImg)
    return pupilEllipses 



# ------------------- PART 2
def detectPupilKMeans(gray,K,distanceWeight,reSize):
    ''' Detects the pupil in the image, gray, using k-means
gray : grays scale image
K : Number of clusters
distanceWeight : Defines the weight of the position parameters
reSize : the size of the image to do k-means on
'''
    #Resize for faster performance
    smallI = cv2.resize(gray, reSize)
    M,N = smallI.shape
    #Generate coordinates in a matrix
    X,Y = np.meshgrid(range(M),range(N))
    #Make coordinates and intensity into one vectors
    z = smallI.flatten()
    x = X.flatten()
    y = Y.flatten()
    O = len(x)
    #make a feature vectors containing (x,y,intensity)
    features = np.zeros((O,3))
    features[:,0] = z;
    features[:,1] = y/distanceWeight; #Divide so that the distance of position weighs lessthan intensity
    features[:,2] = x/distanceWeight;
    features = np.array(features,'f')
    # cluster data
    centroids,variance = kmeans(features,K)
    centroids.sort(axis = 0) # Sorting clusters according to intensity (ascending)
    pupilCluster = centroids[0] #Choosing the lowest intesity cluster. pupilCluster[0] is threshold for finding pupil for later
    
    
    
    #Wiktor's way of BLOB detection which is not that accurate-----------------------------------
    tempResultImg = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
    val,binI = cv2.threshold(gray, pupilCluster[0], 255, cv2.THRESH_BINARY_INV)
    
    #Appplying morphology
    st7 = cv2.getStructuringElement(cv2.MORPH_CROSS,(7,7))
    st15 = cv2.getStructuringElement(cv2.MORPH_CROSS,(15,15))
             
    binI = cv2.morphologyEx(binI, cv2.MORPH_CLOSE, st15) #Close 
    binI= cv2.morphologyEx(binI, cv2.MORPH_OPEN, st7)
   
    #binI = cv2.morphologyEx(binI, cv2.MORPH_DILATE, st7, iterations=2) #Dialite 
    
    cv2.imshow("ThresholdPupil",binI)
     
    sliderVals = getSliderVals() #Getting slider values
    props = RegionProps() 
    contours, hierarchy = cv2.findContours(binI, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) #Finding contours/candidates for pupil blob
    pupils = []
    pupilEllipses = []
    for cnt in contours:
        values = props.CalcContourProperties(cnt,['Area','Length','Centroid','Extend','ConvexHull']) #BUG - Add cnt.astype('int') in Windows
        if values['Area'] < sliderVals['maxSizePupil'] and values['Area'] > sliderVals['minSizePupil'] and values['Extend'] < 0.9:
            pupils.append(values)
            centroid = (int(values['Centroid'][0]),int(values['Centroid'][1]))
            cv2.circle(tempResultImg,centroid, 2, (0,0,255),4)
            pupilEllipses.append(cv2.fitEllipse(cnt))
    cv2.imshow("TempResults",tempResultImg)
    #return pupilEllipses 
    
 #--------------------------------------This snippet belongs to function detectPupilKmeans if there were any doubts-------   
    #use the found clusters to map
    label,distance = vq(features,centroids)
    # re-create image from
    labelIm = np.array(np.reshape(label,(M,N)))
    
    '''This snippet is my try of applying BLOB detection on labelIm (ex 1.7) . I give up and I see no sense in doing that to be honest. Maybe I don't unerstand that.'''
    #Very ugly way of extracting pupil cluster on labelIm. It can be done in two lines I'm sure. I have no Idea why they made us do blob detection on labelIm anyway. I can have perfectly fine result on gray image using data I laready have
    #labelCopy = []
    #for a in labelIm:
    #    newBlock = []
    #    for b in a:
    #        if b !=0: 
    #            b=255
    #        newBlock.append(b)
    #    labelCopy.append(newBlock)
    
    #Applying some morphology
    #labelCopy = np.array(labelCopy)
    #Here I get binary image showing only cluster containing pixels which intensity equals pupil intensity. Here we should continue with blob detection...     
    ''' end snippet'''
    
    f = figure(1)
    imshow(labelIm) #labelIm
    f.canvas.draw()
    f.show()
 #-------------------------------------------------------------------------------------------   
         

def getPupilThershold(gray, K, distanceWeight):
     #Resize for faster performance
    smallI = cv2.resize(gray, (40,40))
    M,N = smallI.shape
    #Generate coordinates in a matrix
    X,Y = np.meshgrid(range(M),range(N))
    #Make coordinates and intensity into one vectors
    z = smallI.flatten()
    x = X.flatten()
    y = Y.flatten()
    O = len(x)
    #make a feature vectors containing (x,y,intensity)
    features = np.zeros((O,3))
    features[:,0] = z;
    features[:,1] = y/distanceWeight; #Divide so that the distance of position weighs less than intensity
    features[:,2] = x/distanceWeight;
    features = np.array(features,'f')
    # cluster data
    centroids,variance = kmeans(features,K)
    centroids.sort(axis = 0) # Sorting clusters according to intensity (ascending)
    pupilCluster = centroids[0] #Choosing the lowest intesity cluster. pupilCluster[0] is threshold for finding pupil
    return pupilCluster[0]
    


def GetGlints(gray,thr):
    tempResultImg = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR) #used to draw temporary results

    props = RegionProps()
    val,binI = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY) #Using non inverted binary image
    
    #Combining opening and dialiting seems to be the best but is it ok that other glints are visible two?????!!!!!
    st7 = cv2.getStructuringElement(cv2.MORPH_CROSS,(7,7))
    st9 = cv2.getStructuringElement(cv2.MORPH_CROSS,(7,7))
    
    binI= cv2.morphologyEx(binI, cv2.MORPH_OPEN, st7)
    binI = cv2.morphologyEx(binI, cv2.MORPH_DILATE, st9, iterations=2)
    
    cv2.imshow("ThresholdGlints",binI)
    #Calculate blobs
    sliderVals = getSliderVals() #Getting slider values
    contours, hierarchy = cv2.findContours(binI, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) #Finding contours/candidates for pupil blob
    glints = []
    glintEllipses = []
    for cnt in contours:
        values = props.CalcContourProperties(cnt,['Area','Length','Centroid','Extend','ConvexHull']) #BUG - Add cnt.astype('int') in Windows
        if values['Area'] < sliderVals['maxSizeGlints'] and values['Area'] > sliderVals['minSizeGlints']:
            glints.append(values)
            centroid = (int(values['Centroid'][0]),int(values['Centroid'][1]))
            cv2.circle(tempResultImg,centroid, 2, (0,0,255),4)
            glintEllipses.append(cv2.fitEllipse(cnt))
    cv2.imshow("TempResults",tempResultImg)
    return glintEllipses


def GetIrisUsingThreshold(gray, thr):
    #
    #NOTE: Detecting Iris uses GlintsDetection UI for adjusting parameters. You eather run glints or iris detection in 'update' method.
    #
    #Assignment 1 part2 (2.1)
    #It is almost impossible to detect iris using threshold because I can't get threshold so that Iris becomes an ellipse. It always looks like croissant ;P.  
    #
    tempResultImg = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR) #used to draw temporary results

    props = RegionProps()
    val,binI = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY) #Using non inverted binary image
    
    #
    #Applying morphology
    #Nothing seems to work here!!
    st7 = cv2.getStructuringElement(cv2.MORPH_CROSS,(13,13))
    st9 = cv2.getStructuringElement(cv2.MORPH_CROSS, (7,7))
    
    binI= cv2.morphologyEx(binI, cv2.MORPH_OPEN, st7)
    #binI= cv2.morphologyEx(binI, cv2.MORPH_CLOSE, st7)
    #binI = cv2.morphologyEx(binI, cv2.MORPH_DILATE, st9, iterations=2)
    
    cv2.imshow("ThresholdGlints",binI)
    #Calculate blobs
    sliderVals = getSliderVals() #Getting slider values
    contours, hierarchy = cv2.findContours(binI, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) #Finding contours/candidates for pupil blob
    irises = []
    irisEllipses = []
    for cnt in contours:
        values = props.CalcContourProperties(cnt,['Area','Length','Centroid','Extend','ConvexHull']) #BUG - Add cnt.astype('int') in Windows
        if values['Area'] < sliderVals['maxSizeGlints'] and values['Area'] > sliderVals['minSizeGlints']:
            irises.append(values)
            centroid = (int(values['Centroid'][0]),int(values['Centroid'][1]))
            cv2.circle(tempResultImg,centroid, 2, (0,0,255),4)
            irisEllipses.append(cv2.fitEllipse(cnt))
    cv2.imshow("TempResults",tempResultImg)
    return irisEllipses

def GetGradientImageInfo(I):

    #
    #Mandatory Assignment 1 2.2 (1)
    #
    #Creating gradient X and Y images.
    gray = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
    
    kernelSobelX = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])/9.0
    kernelSobelY = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])/9.0
    
    DstSobelX = cv2.filter2D(gray, -1,kernelSobelX)
    DstSobelY = cv2.filter2D(gray, -1,kernelSobelY)
    
    #Orientation
    OI = np.arctan2(DstSobelX,DstSobelY)*180/math.pi
    
    #Magnitude
    DeltaI = np.sqrt(np.power(DstSobelY,2) + np.power(DstSobelX,2))
    
    
    #Just to see how they look like :-)
    #
    #fx = figure("Gradient image test")
    #imshow(DstSobelX)
    #fx.canvas.draw()
    #fx.show()
    
    
def circleTest(nPts,C,circleRadius):
    P = getCircleSamples(center=C, radius=circleRadius, nPoints=nPts)
    return P
    

def circularHough(gray):
    ''' Performs a circular hough transform of the image, gray and shows the  detected circles
    The circe with most votes is shown in red and the rest in green colors '''
    #See help for http://opencv.itseez.com/modules/imgproc/doc/feature_detection.html?highlight=houghcircle#cv2.HoughCircles
    blur = cv2.GaussianBlur(gray, (31,31), 11)

    dp = 6; minDist = 30
    highThr = 20 #High threshold for canny
    accThr = 850 #accumulator threshold for the circle centers at the detection stage. The smaller it is, the more false circles may be detected
    maxRadius = 50
    minRadius = 155
    circles = cv2.HoughCircles(blur,cv2.cv.CV_HOUGH_GRADIENT, dp,minDist, None, highThr,accThr,maxRadius, minRadius)

    #Make a color image from gray for display purposes
    gColor = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if (circles !=None):
    #print circles
        all_circles = circles[0]
        M,N = all_circles.shape
        k=1
        for c in all_circles:
            cv2.circle(gColor, (int(c[0]),int(c[1])),c[2], (int(k*255/M),k*128,0))
            k=k+1
    c=all_circles[0,:]
    cv2.circle(gColor, (int(c[0]),int(c[1])),c[2], (0,0,255),5)
    cv2.imshow("hough",gColor)
    


def GetIrisUsingNormals(gray,pupil,normalLength):
    ''' Given a gray level image, gray and the length of the normals, normalLength
    return a list of iris locations'''
    # YOUR IMPLEMENTATION HERE !!!!
    pass

def GetIrisUsingSimplifyedHough(gray,pupil):
    ''' Given a gray level image, gray
    return a list of iris locations using a simplified Hough transformation'''
    # YOUR IMPLEMENTATION HERE !!!!
    pass

def GetEyeCorners(leftTemplate, rightTemplate,pupilPosition=None):
    pass

def FilterPupilGlint(pupils,glints):
    ''' Given a list of pupil candidates and glint candidates returns a list of pupil and glints'''
    pass

def update(I):
    '''Calculate the image features and display the result based on the slider values'''
    #global drawImg
    global frameNr,drawImg
    img = I.copy()
    sliderVals = getSliderVals()
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    # Do the magic
    pupils = []#GetPupil(gray, sliderVals["pupilThr"]) #Get pupil manual threshold
    #pupils = GetPupil(gray,getPupilThershold(gray, K=4, distanceWeight=2)) #Automatic thresholding using Kmeans. Could be quite precise when using Atanas'es pupil filtering.
    glints = []#GetGlints(gray,sliderVals['glintThr'])
    #irises = GetIrisUsingThreshold(gray, sliderVals['glintThr'])
    
    #FilterPupilGlint(pupils, glints)
    
    #Assignment 1 part 2|| 2.2 (1)
    #
    #GetGradientImageInfo(I)
    #
    
    
    #-----------------------Detect pupil K-Means
    
     #
    #Assignemtn 1 part 2 || 1 (2,3) 
    #Changing the distanceWeight doesnt seem to have any influence on pupil detection. Am I doing something wrong? 
    #I guess its becuase we have too few clusters.
    # 
    distanceWeight = 2   #used wen running only detectPupilKmeans //good value 2                  
    
    
    #
    #Assignment 1 part 2 || 1 (4)
    #Values K = 4 and 5 separates pupil the best. When the values are highier than 5 there are too many classes for classification and output image gets more eroded and pupil less visible . 
    #Moreover if we keep value at 4 we can even detect Iris but its not properly classified on every frame.
    #
    K = 4 #used wen running only detectPupilKmeans //good value 4
    
    #
    #Assignment 1 part 2 || 1 (5)
    #The values that give the best values are K=4 and distanceWeight=2 (making it higher for low amount of clusters makes almost no difference).  
    #This values doesn't apply that well in other sequences but they are still batter than in binary thresholding.
    # 
  
    reSize = (40,40) #used wen running only detectPupilKmeans
    
    detectPupilKMeans(gray, K, distanceWeight, reSize) 
    
    #
    #Assignment 1 part 2 || 1 (8)
    #The advantage of using clustering before BLOB detection is that we can find pixel intensity of the object that we are looking for and therefore automatically set threshold.  
    #This isn't however an ultimate method. Using clustering still requires setting number of cluster and distance weight between pixels. The more diverse lighting conditions the more clusters you have to use. 
    #
    
    #
    #Assignment 1 part 2 || 1 (10)
    #Not in this example. In some very rare condiitons (not with eye tracking) you could definately do that because when you use Kmeans you get center points of clusters whoch you could use. 
    #If every object had different pixel intensity then yes, you could use Kmeans for detecting some objects. 
    #
    
    #-----------------------------------------------
    
    #Do template matching
    global leftTemplate
    global rightTemplate
    GetEyeCorners(leftTemplate, rightTemplate)
    #Display results
    global frameNr,drawImg
    x,y = 15,10
    setText(img,(520,y+10),"Frame:%d" %frameNr)
    sliderVals = getSliderVals()

    # for non-windows machines we print the values of the threshold in the original image
    if sys.platform != 'win32':
        step=18
        cv2.putText(img, "pupilThr :"+str(sliderVals['pupilThr']), (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.CV_AA)
        cv2.putText(img, "glintThr :"+str(sliderVals['glintThr']), (x, y+step), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.CV_AA)
        cv2.putText(img, "maxSizePupil :"+str(sliderVals['maxSizePupil']), (x, y+2*step), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.CV_AA)
        cv2.putText(img, "minSizePupil :"+str(sliderVals['minSizePupil']), (x, y+3*step), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.CV_AA)
        cv2.putText(img, "maxSizeGlints :"+str(sliderVals['maxSizeGlints']), (x, y+4*step), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.CV_AA)
        cv2.putText(img, "minSizeGlints :"+str(sliderVals['minSizeGlints']), (x, y+5*step), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.CV_AA)
        

    for pupil in pupils:
        cv2.ellipse(img,pupil,(0,255,0),1)
        C = int(pupil[0][0]),int(pupil[0][1])
        cv2.circle(img,C, 2, (0,0,255),4)
        
        #Assignment 1 part 2|| 2.2 (5)
        #Getting circle points around iris for every pupil.
        
        #P = circleTest(20, C, 70)
        #for circlePoint in P:
        #    circlePointCoor = (int(circlePoint[0]), int(circlePoint[1]))
        #    cv2.circle(img, circlePointCoor, 1, (255,243,17), 4)#Drawing points around pupil centroid
        #    #Assignment 1 part 2|| 2.2 (6)
        #    #Drawing lines through points (not sure if it is correct)
        #    gradientPoint = (int(circlePoint[0]+circlePoint[2]*20), int(circlePoint[1]+circlePoint[3]*20)) #Generating gradient points to connect the lines. Multiplied by 20 to make them longer.
        #    cv2.line(img,circlePointCoor, gradientPoint,(13,243,17), 1)#Drawing lines form circlePoints though gradient points.
        
    for glint in glints:
        C = int(glint[0][0]),int(glint[0][1])
        cv2.circle(img,C, 2,(255,0,255),5)
    cv2.imshow("Result", img)

        #For Iris detection - Week 2
        #circularHough(gray)

    #copy the image so that the result image (img) can be saved in the movie
    drawImg = img.copy()


def printUsage():
    print "Q or ESC: Stop"
    print "SPACE: Pause"
    print "r: reload video"
    print 'm: Mark region when the video has paus ed'
    print 's: toggle video  writing'
    print 'c: close video sequence'

def run(fileName,resultFile='eyeTrackingResults.avi'):

    ''' MAIN Method to load the image sequence and handle user inputs'''
    global imgOrig, frameNr,drawImg
    setupWindowSliders()
    props = RegionProps()
    cap,imgOrig,sequenceOK = getImageSequence(fileName)
    videoWriter = 0

    frameNr =0
    if(sequenceOK):
        update(imgOrig)
    printUsage()
    frameNr=0
    saveFrames = False

    while(sequenceOK):
        sliderVals = getSliderVals()
        frameNr=frameNr+1
        ch = cv2.waitKey(1)
        #Select regions
        if(ch==ord('m')):
            if(not sliderVals['Running']):
                roiSelect=ROISelector(imgOrig)
                pts,regionSelected= roiSelect.SelectArea('Select left eye corner',(400,200))
                if(regionSelected):
                    leftTemplate = imgOrig[pts[0][1]:pts[1][1],pts[0][0]:pts[1][0]]

        if ch == 27:
            break
        if (ch==ord('s')):
            if((saveFrames)):
                videoWriter.release()
                saveFrames=False
                print "End recording"
            else:
                imSize = np.shape(imgOrig)
                videoWriter = cv2.VideoWriter(resultFile, cv.CV_FOURCC('D','I','V','3'), 15.0,(imSize[1],imSize[0]),True) #Make a video writer
                saveFrames = True
                print "Recording..."



        if(ch==ord('q')):
            break
        if(ch==32): #Spacebar
            sliderVals = getSliderVals()
            cv2.setTrackbarPos('Stop/Start','ThresholdGlints',not sliderVals['Running'])
            cv2.setTrackbarPos('Stop/Start','ThresholdPupil',not sliderVals['Running'])
        if(ch==ord('r')):
            frameNr =0
            sequenceOK=False
            cap,imgOrig,sequenceOK = getImageSequence(fileName)
            update(imgOrig)
            sequenceOK=True

        sliderVals=getSliderVals()
        if(sliderVals['Running']):
            sequenceOK, imgOrig = cap.read()
            if(sequenceOK): #if there is an image
                update(imgOrig)
            if(saveFrames):
                videoWriter.write(drawImg)

    videoWriter.release



#--------------------------
#         UI related
#--------------------------

def setText(dst, (x, y), s):
    cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv2.CV_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.CV_AA)


def setupWindowSliders():
    cv2.namedWindow("Result")
    cv2.namedWindow('ThresholdPupil')
    cv2.namedWindow('ThresholdGlints')
    cv2.namedWindow("TempResults")
    #Threshold value for the pupil intensity
    cv2.createTrackbar('pupilThr','ThresholdPupil', 90, 255, onSlidersChange)
    #Threshold value for the glint intensities
    cv2.createTrackbar('glintThr','ThresholdGlints', 245, 255,onSlidersChange)
    #define the minimum and maximum areas of the pupil
    cv2.createTrackbar('minSizePupil','ThresholdPupil', 20, 200, onSlidersChange)
    cv2.createTrackbar('maxSizePupil','ThresholdPupil', 200,200, onSlidersChange)
    #define the minimum and maximum areas of the glints
    cv2.createTrackbar('minSizeGlints','ThresholdGlints', 10, 50, onSlidersChange)
    cv2.createTrackbar('maxSizeGlints','ThresholdGlints', 50,50, onSlidersChange)
    #Value to indicate whether to run or pause the video
    cv2.createTrackbar('Stop/Start','ThresholdPupil', 0,1, onSlidersChange)
    cv2.createTrackbar('Stop/Start','ThresholdGlints', 0,1, onSlidersChange)

def getSliderVals():
    '''Extract the values of the sliders and return these in a dictionary'''
    sliderVals={}
    sliderVals['pupilThr'] = cv2.getTrackbarPos('pupilThr', 'ThresholdPupil')
    sliderVals['glintThr'] = cv2.getTrackbarPos('glintThr', 'ThresholdGlints')
    sliderVals['minSizePupil'] = 50*cv2.getTrackbarPos('minSizePupil', 'ThresholdPupil')
    sliderVals['maxSizePupil'] = 50*cv2.getTrackbarPos('maxSizePupil', 'ThresholdPupil')
    sliderVals['minSizeGlints'] = 50*cv2.getTrackbarPos('minSizeGlints', 'ThresholdGlints')
    sliderVals['maxSizeGlints'] = 50*cv2.getTrackbarPos('maxSizeGlints', 'ThresholdGlints')
    sliderVals['Running'] = 1==cv2.getTrackbarPos('Stop/Start', 'ThresholdPupil')
    sliderVals['Running'] = 1==cv2.getTrackbarPos('Stop/Start', 'ThresholdGlints')
    return sliderVals

def onSlidersChange(dummy=None):
    ''' Handle updates when slides have changed.
     This  function only updates the display when the video is put on pause'''
    global imgOrig
    sv=getSliderVals()
    if(not sv['Running']): # if pause
        update(imgOrig)

#--------------------------
#         main
#--------------------------
run(inputFile)