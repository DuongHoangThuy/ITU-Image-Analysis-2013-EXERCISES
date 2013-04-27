from scipy import ndimage
import cv2
import numpy as n
from pylab import *
from matplotlib import *
from matplotlib.pyplot import *
from scipy import *
import math
import SIGBTools
from numpy.oldnumeric.compat import matrixmultiply


projectResources = '/Users/wzdziechowski/Desktop/IA2Resources/'


def frameTrackingData2BoxData(data):
    #Convert a row of points into tuple of points for each rectangle
    pts= [ (int(data[i]),int(data[i+1])) for i in range(0,11,2) ]
    boxes = [];
    for i in range(0,7,2):
        box = tuple(pts[i:i+2])
        boxes.append(box)   
    return boxes


def simpleTextureMap():

    I1 = cv2.imread(projectResources + 'Images/ITULogo.jpg')
    I2 = cv2.imread(projectResources + 'Images/ITUMap.bmp')

    #Print Help
    H,Points  = SIGBTools.getHomographyFromMouse(I1,I2,4)
    h, w,d = I2.shape
    overlay = cv2.warpPerspective(I1, H,(w, h))
    M = cv2.addWeighted(I2, 0.5, overlay, 0.5,0)

    cv2.imshow("Overlayed Image",M)
    cv2.waitKey(0)

def showImageandPlot(N):
    #A simple attenmpt to get mouse inputs and display images using matplotlib
    I = cv2.imread('groundfloor.bmp')
    drawI = I.copy()
    #make figure and two subplots
    fig = figure(1) 
    ax1  = subplot(1,2,1) 
    ax2  = subplot(1,2,2) 
    ax1.imshow(I) 
    ax2.imshow(drawI)
    ax1.axis('image') 
    ax1.axis('off') 
    points = fig.ginput(5) 
    fig.hold('on')
    
    for p in points:
        #Draw on figure
        subplot(1,2,1)
        plot(p[0],p[1],'rx')
        #Draw in image
        cv2.circle(drawI,(int(p[0]),int(p[1])),2,(0,255,0),10)
    ax2.cla
    ax2.imshow(drawI)
    draw() #update display: updates are usually defered 
    show()
    savefig('somefig.jpg')
    cv2.imwrite("drawImage.jpg", drawI)


def texturemapGridSequence():
    """ Skeleton for texturemapping on a video sequence"""
    fn = projectResources + 'GridVideos/grid1.mp4'
    cap = cv2.VideoCapture(fn)
    drawContours = True;

    texture = cv2.imread(projectResources + 'Images/ITULogo.jpg')
    texture = cv2.pyrDown(texture)
    
    #Points that will be used to colaculate Homography
    imagePoints = [] 
    
    #Getting texture corner points
    m,n,d = texture.shape
    imagePoints.append([(float(0.0),float(0.0)),(float(n),0),(float(n),float(m)),(0,m)])
    
    print imagePoints


    #mTex,nTex,t = texture.shape

    #load Tracking data
    running, imgOrig = cap.read()
    mI,nI,t = imgOrig.shape

    cv2.imshow("win2",imgOrig)

    pattern_size = (9, 6)

    idx = [0,8,45,53]
    while(running):
    #load Tracking data
        running, imgOrig = cap.read()
        if(running):
            imgOrig = cv2.pyrDown(imgOrig)
            gray = cv2.cvtColor(imgOrig,cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, pattern_size)
            if found:
                term = ( cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1 )
                cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), term)
                cv2.drawChessboardCorners(imgOrig, pattern_size, corners, found)
                print "BLOB "
                print corners

                for t in idx:
                    cv2.circle(imgOrig,(int(corners[t,0,0]),int(corners[t,0,1])),10,(255,t,t))
            cv2.imshow("win2",imgOrig)
            cv2.waitKey(1)



def realisticTexturemap(scale,point,map):
    I1 = map
    running, I2 = (cv2.VideoCapture(projectResources + 'GroundFloorData/SunClipDS.mp4')).read()
    I3 = cv2.imread(projectResources + 'Images/ITULogo.jpg')
    (x,y)=point

    H_G_M = np.matrix(np.load('Results/H_G_M.npy'))
    H_T_M = np.matrix([[scale,0,x],[0,scale,y],[0,0,1]]) #no rotation
    H_T_G = H_G_M.I * H_T_M
    h, w, d = I2.shape
    overlay = cv2.warpPerspective(I3, H_T_G ,(w, h))
    M = cv2.addWeighted(I2, 0, overlay, 1,0)
    cv2.imshow('H_T_G', M)
    cv2.waitKey(40000)

    #print "Not implemented yet\n"*30
    return None


def showFloorTrackingData():
    #Load videodata
    fn = projectResources + "GroundFloorData/sunclipds.avi"
    cap = cv2.VideoCapture(fn)
    
    #load Tracking data
    running, imgOrig = cap.read()
    dataFile = np.loadtxt(projectResources + 'GroundFloorData/trackingdata.dat')
    m,n = dataFile.shape
    
    fig = figure()
    for k in range(m):
        running, imgOrig = cap.read() 
        if(running):
            boxes= frameTrackingData2BoxData(dataFile[k,:])
            boxColors = [(255,0,0),(0,255,0),(0,0,255)]
            for k in range(0,3):
                aBox = boxes[k]
                cv2.rectangle(imgOrig, aBox[0], aBox[1], boxColors[k])
            cv2.imshow("boxes",imgOrig);
            cv2.waitKey(1)

def angle_cos(p0, p1, p2):
    d1, d2 = p0-p1, p2-p1
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

def findSquares(img,minSize = 2000,maxAngle = 1):
    """ findSquares intend to locate rectangle in the image of minimum area, minSize, and maximum angle, maxAngle, between 
    sides"""
    squares = []
    contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
         cnt_len = cv2.arcLength(cnt, True)
         cnt = cv2.approxPolyDP(cnt, 0.08*cnt_len, True)
         if len(cnt) == 4 and cv2.contourArea(cnt) > minSize and cv2.isContourConvex(cnt):
             cnt = cnt.reshape(-1, 2)
             max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in xrange(4)])
             if max_cos < maxAngle:
                 squares.append(cnt)
    return squares

def DetectPlaneObject(I,minSize=1000):
      """ A simple attempt to detect rectangular 
      color regions in the image"""
      HSV = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
      h = HSV[:,:,0].astype('uint8')
      s = HSV[:,:,1].astype('uint8')
      v = HSV[:,:,2].astype('uint8')
      
      b = I[:,:,0].astype('uint8')
      g = I[:,:,1].astype('uint8')
      r = I[:,:,2].astype('uint8')
     
      # use red channel for detection.
      s = (255*(r>230)).astype('uint8')
      iShow = cv2.cvtColor(s, cv2.COLOR_GRAY2BGR)
      cv2.imshow('ColorDetection',iShow)
      squares = findSquares(s,minSize)
      return squares
  
def texturemapObjectSequence():
    """ Poor implementation of simple texturemap """
    fn = projectResources + 'BookVideos/Seq3_scene.mp4'
    cap = cv2.VideoCapture(fn) 
    drawContours = True;
    
    texture = cv2.imread(projectResources + 'images/ITULogo.jpg')
    #texture = cv2.transpose(texture)
    mTex,nTex,t = texture.shape
    
    #load Tracking data
    running, imgOrig = cap.read()
    mI,nI,t = imgOrig.shape
    
    print running 
    while(running):
        for t in range(20):
            running, imgOrig = cap.read() 
        
        if(running):
            squares = DetectPlaneObject(imgOrig)
            
            for sqr in squares:
                 #Do texturemap here!!!!
                 #TODO
                 
                 if(drawContours):                
                     for p in sqr:
                         cv2.circle(imgOrig,(int(p[0]),int(p[1])),3,(255,0,0)) 
                 
            
            if(drawContours and len(squares)>0):    
                cv2.drawContours( imgOrig, squares, -1, (0, 255, 0), 3 )

            cv2.circle(imgOrig,(100,100),10,(255,0,0))
            cv2.imshow("Detection",imgOrig)
            cv2.waitKey(1)
#showFloorTrackingData()
#simpleTextureMap()
#realisticTexturemap(0,0,0)
#texturemapGridSequence()
#showFloorTrackingData()

''' -------------------------------------------------------------- Our methods '''

def DisplayTraceSatic(homography):
    trackingData = loadtxt(projectResources + "GroundFloorData/trackingdata.dat")
    trackingData = SIGBTools.toHomogenious(trackingData)
    #rotationCW90 = n.matrix([[n.cos(90), n.sin(90), 0],[-n.sin(90), n.cos(90) ,0],[0,0,1]])
    #homography = homography * rotationCW90
    
    transformedPoints = list()
    
    for p in trackingData:
        currentPoint = n.matrix([p[2], p[3],1]).T
        transformedPoints.append(matrixmultiply(homography, currentPoint))
        
    I = cv2.imread(projectResources + "Images/ITUMap.bmp")
    drawI = I.copy()
    fig = figure(1) 
    ax1  = subplot(1,2,1) 
    ax2  = subplot(1,2,2) 
    ax1.imshow(I) 
    ax2.imshow(drawI)
    ax1.axis('image') 
    ax1.axis('off')  
    fig.hold('on')
    
    for p in transformedPoints:
        subplot(1,2,1)
        plot(p[0],p[1],'rx')
        cv2.circle(drawI,(int(p[0]),int(p[1])),2,(0,255,0),1)
    
    ax2.cla
    ax2.imshow(drawI)
    draw() #update display: updates are usually defered 
    show()
    cv2.imwrite("drawImage.jpg", drawI)
    
    return transformedPoints

def DispalyTraceDynamic(homography):
    sequence = projectResources + "GroundFloorData/sunclipds.avi"
    sequence = cv2.VideoCapture(fn)
    
    floorPlan = cv2.imread(projectResources + "Images/ITUMap.bmp")
    
    trackingData = loadtxt(projectResources + "GroundFloorData/trackingdata.dat")
    trackingData = SIGBTools.toHomogenious(trackingData)
    m,n = trackingData.shape
    
    drawI = floorPlan.copy()
    running, cap  = sequence.read()
    
    for p in trackingData:
        running, cap = sequence.read();
        cv2.imshow("Sequence", cap)
        cv2.waitKey(1)
        if(running):
            currentPoint = matrix([p[2], p[3],1]).T
            hPoint = matrixmultiply(homography, currentPoint)
            cv2.circle(drawI,(int(hPoint[0]),int(hPoint[1])),2,(0,255,0),1)
            cv2.imshow("Plan", drawI)
            cv2.waitKey(1)
            #print hPoint
    return

#MIND HOW YUU SELECT THE POINTS!!!!! Start form upper left corner and go CW
def texturemapGroundFloor():
    sequence = projectResources + "GroundFloorData/sunclipds.avi"
    texture = cv2.imread(projectResources + 'Images/ITULogo.jpg')
    sequence = cv2.VideoCapture(sequence)
    texture = texture.copy()
    running, firstFrame = sequence.read();
    H, points = SIGBTools.getHomographyFromMouse(texture,firstFrame,-4)
    h, w, d = firstFrame.shape
    overlay = cv2.warpPerspective(cv2.transpose(texture), H,(w, h))
    while(True):
        if(running):
            running, cap = sequence.read()
            M = cv2.addWeighted(cap, 1.0, overlay, 0.5,0)
            cv2.imshow("Texture", M)
            cv2.waitKey(1)
    return

def playVideo():
    sequence = "/Users/wzdziechowski/Desktop/chessSequence.mov"
    sequence = cv2.VideoCapture(sequence)
    while(True):
        running, cap = sequence.read();
        cv2.imshow("Sequence", cap)
        cv2.waitKey(1)
    return
        
      


''' ---------------------------------------------------------------- Runnings '''


''' Estimating homography for point transformation '''
#fn = projectResources + "GroundFloorData/sunclipds.avi"
#cap = cv2.VideoCapture(fn)
#running, imgOrig = cap.read()
#floorPlan = cv2.imread(projectResources + "Images/ITUMap.bmp")
#print SIGBTools.getHomographyFromMouse(imgOrig, floorPlan)

homography1 = array([[ -4.34489218e-02,   7.83575801e-01,   1.45998418e+02],
       [ -6.66450384e-01,   2.45967210e-02,   2.14616412e+02],
       [ -6.78926807e-04,  -2.35062147e-04,   1.00000000e+00]])#Best so far



#DispalyTraceDynamic(homography1)

#showFloorTrackingData()

#playVideo()

#simpleTextureMap()

#texturemapGroundFloor()

#texturemapGridSequence()

#map = cv2.imread(projectResources +'Images/ITUMap.bmp')
#fig = figure(1)
#ax1 = subplot(1,2,1)
#ax1.imshow(map)
#point = fig.ginput(1)

#realisticTexturemap(0.5,point[0],map)


playVideo()




