import cv2
import cv
import pylab
from SIGBTools import RegionProps
from SIGBTools import getLineCoordinates
from SIGBTools import ROISelector
from SIGBTools import getImageSequence
import numpy as np
import sys
import math




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

#       Blur the image (with a gausian filter, but ideally box filter (all 1s)), then apply histogram equalization to increase the contrast
#+/-:   threshold works better, but relying on the angle of ellipses decreases
#       grayBlured=cv2.GaussianBlur(gray, (41,41),0)
#       grayBlured=cv2.equalizeHist(grayBlured)

#       Do inverted thresholding:
    props = RegionProps()
    val,binI = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY_INV)#use grayBlurred?

#       Do morphological operations close, open and dialite(to increase the area size from opening)
#+/-:   remove noise or small objects, but remaining objects will be rounder and relying on the angle of the ellipses decreases
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
#       Gather the remaining blobs for analysis
    contours, hierarchy = cv2.findContours(binI, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) #Finding contours/candidates for pupil blob
    pupils = []#used to store center of blobs
    pupilEllipses = []#used to store ellipses
    for cnt in contours:
        if len(cnt)>=5:

            values = props.CalcContourProperties(cnt,['Area','Length','Centroid','Perimiter','Extend','ConvexHull']) #BUG - Add cnt.astype('int') in Windows
#       Calculate circularity to eliminate pupil candidates... unfortunately, the perimeter using a function from SIGBTools seems inaccurate and calculating it might be hard, therefore improvise by using the angle of the ellipse
#       circularity=values['Perimiter']/(2*math.sqrt(math.pi*values['Area']))
            #print circularity
#       Calculate the ellipse the blob best fits in using cv2 tools
            ellipse=cv2.fitEllipse(cnt.astype('int'))# fitEllipse([center][height,width],[angle])
            #print ellipse,ellipse[2]
#       Filter blobs both on area size and angle of the ellipse
#+/-    we filter ellipses with angle between 80-100 degrees, but this doesn't mean circularity (e.g. a blob in the form of a cat/snake eye has also 90 degrees), but it suffices in our case... even circularity doesn't mean 100% correctness
            if values['Area'] < sliderVals['maxSizePupil'] and values['Area'] > sliderVals['minSizePupil'] and ellipse[2]>80 and ellipse[2]<100:
                pupils.append(values)
                centroid = (int(values['Centroid'][0]),int(values['Centroid'][1]))
                pupilEllipses.append(ellipse)
    cv2.imshow("TempResults",tempResultImg)

    return pupilEllipses


def GetGlints(gray,thr):
    tempResultImg = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR) #used to draw temporary results

    props = RegionProps()

#       Do inverted thresholding:
    val,binI = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY) #Using non inverted binary image

#       Do morphological operations open, dilate(to increase the area size from opening)
    st7 = cv2.getStructuringElement(cv2.MORPH_CROSS,(7,7))
    st9 = cv2.getStructuringElement(cv2.MORPH_CROSS,(7,7))
    
    binI= cv2.morphologyEx(binI, cv2.MORPH_OPEN, st7)
    binI = cv2.morphologyEx(binI, cv2.MORPH_DILATE, st9, iterations=2)
    
    cv2.imshow("ThresholdGlints",binI)

#       Calculate blobs
    sliderVals = getSliderVals() #Getting slider values
    contours, hierarchy = cv2.findContours(binI, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) #Finding contours/candidates for pupil blob
    glints = []
    glintEllipses = []
    for cnt in contours:
        values = props.CalcContourProperties(cnt.astype('int'),['Area','Length','Centroid','Extend','ConvexHull']) #BUG - Add cnt.astype('int') in Windows
        if len(cnt)>=5:
#       Calculate the ellipse the blob best fits in using cv2 tools
            ellipse=cv2.fitEllipse(cnt.astype('int'))# fitEllipse([center][height,width],[angle])
#       Filter blobs on area size
#+/-:   we haven't found the actual glints, but we get a very good list of candidates
        if values['Area'] < sliderVals['maxSizeGlints'] and values['Area'] > sliderVals['minSizeGlints']:
            glints.append(values)
            centroid = (int(values['Centroid'][0]),int(values['Centroid'][1]))
            #cv2.circle(tempResultImg,centroid, 2, (0,0,255),4)
            glintEllipses.append(cv2.fitEllipse(cnt.astype('int')))
    cv2.imshow("TempResults",tempResultImg)
    return glintEllipses

def FilterPupilGlint(pupils,glints):
    ''' Given a list of pupil candidates and glint candidates returns a list of pupil and glints'''
#       Filter glints based on fixed distance between them - impossible in a single frame... need a previous frame, but this is not build for that... it is much harder (time consuming for computation) than simply finding the 2 glints that are closest to the pupil
#       Assume that we have found the pupil correctly, we'll find the 2 glints closest to it
#+/-:   for each correct pupil, we are most likely to find the 2 best glint pair candidates (assuming they are among the group of candidates), but it is highly dependent on the pupil/s
    filteredGlints=[]
    #looping through pupilEllipses! / glintEllipses!
    for pupil in range(len(pupils)):
        #finding the 2 min distant glints with 1 loop (need 2 minimums-g1,g2 with values about the current glintEllipse and its distance to the pupil)
        g1,g2=[-1,10**6],[-1,10**6]#[index,min_distance]
        for glint in range(len(glints)):
            if g1[0]==-1:
                g1=[glints[glint],getDistance(pupils[pupil][0],glints[glint][0])]
            elif g2[0]==-1:
                g2=[glints[glint],getDistance(pupils[pupil][0],glints[glint][0])]
            else:
                min=getDistance(pupils[pupil][0],glints[glint][0])
                higher=max(g1[1],g2[1])
                higher_index=1 if higher==g1[1] else 2
                if min<higher:
                    if higher_index==1:
                        g1[0]=glints[glint]
                        g1[1]=min
                    else:
                        g2[0]=glints[glint]
                        g2[1]=min
        if g1[0]!=-1:
            filteredGlints.append(g1[0])
        if g2[0]!=-1:
            filteredGlints.append(g2[0])
    return pupils,filteredGlints

def getDistance(pair1,pair2):
    distance=math.sqrt(math.pow(pair1[0]-pair2[0],2)+math.pow(pair1[1]-pair2[1],2))
    return distance

def GetIrisUsingThreshold(gray,pupil):
    ''' Given a gray level image, gray and threshold
    value return a list of iris locations'''
    # YOUR IMPLEMENTATION HERE !!!!
    pass

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

def update(I):
    '''Calculate the image features and display the result based on the slider values'''
    #global drawImg
    global frameNr,drawImg
    img = I.copy()
    sliderVals = getSliderVals()
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    # Do the magic
    pupils = GetPupil(gray,sliderVals['pupilThr'])
    glints = GetGlints(gray,sliderVals['glintThr'])
    pupils,glints=FilterPupilGlint(pupils,glints)

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
    print 'm: Mark region when the video has paused'
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
                videoWriter = cv2.VideoWriter(resultFile, cv.CV_FOURCC('D','I','V','X'), 15.0,(imSize[1],imSize[0]),True) #Make a video writer
                videoWriter.write(drawImg)
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
    ''' Define windows for displaying the results and create trackbars'''
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