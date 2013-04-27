from scipy import ndimage
import cv2
import cv
import numpy as np
from pylab import *
from matplotlib import *
from matplotlib.pyplot import *
from scipy import *
import math
import SIGBTools
import cubePoints as cube

def calibrationExample():
    camNum =0           # The number of the camera to calibrate
    nPoints = 5        # number of images used for the calibration (space presses)
    patternSize=(9,6)   #size of the calibration pattern
    #saveImage = 'calibrationShoots'

    #calibrated, camera_matrix,dist_coefs,rms = SIGBTools.calibrateCamera(camNum,nPoints,patternSize,saveImage)
    
    #np.save('PMatrix', camera_matrix)
    #np.save('distCoef',dist_coefs)
    
    camera_matrix = np.load('Results/PMatrix.npy')
    dist_coefs = np.load('Results/distCoef.npy')
    calibrated = True
    
    K = camera_matrix
    cam1 = SIGBTools.Camera( np.hstack((K,np.dot(K,np.array([[0],[0],[-1]])) )) )
    cam1.factor()
    #Factor projection matrix into intrinsic and extrinsic parameters
    print "K=",cam1.K
    print "R=",      cam1.R
    print "t",cam1.t
    
    if (calibrated):
        capture = cv2.VideoCapture(camNum)
        running = True
        while running:
            running, img =capture.read()
            imgGray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            ch = cv2.waitKey(1)
            if(ch==27) or (ch==ord('q')): #ESC
                running = False
            img=cv2.undistort(img, camera_matrix, dist_coefs )
            found,corners=cv2.findChessboardCorners(imgGray, patternSize  )
            if (found!=0):
                cv2.drawChessboardCorners(img, patternSize, corners,found)
            cv2.imshow("Calibrated",img)

def frameTrackingData2BoxData(data):
    #Convert a row of points into tuple of points for each rectangle
    pts= [ (int(data[i]),int(data[i+1])) for i in range(0,11,2) ]
    boxes = []
    for i in range(0,7,2):
        box = tuple(pts[i:i+2])
        boxes.append(box)   
    return boxes


def simpleTextureMap():

    I1 = cv2.imread('Images/ITULogo.jpg')
    I2 = cv2.imread('Images/ITUMap.bmp')

    #Print Help
    H,Points  = SIGBTools.getHomographyFromMouse(I1,I2,4)
    h, w,d = I2.shape
    overlay = cv2.warpPerspective(I1, H,(w, h))
    M = cv2.addWeighted(I2, 0.5, overlay, 0.5,0)

    cv2.imshow("Overlayed Image",M)
    cv2.waitKey(0)

def showImageandPlot(N):
    #A simple attempt to get mouse inputs and display images using matplotlib
    I = cv2.imread('Images/ITUMap.bmp')
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
    cap = cv2.VideoCapture('/Users/wzdziechowski/Desktop/IA2Resources/GridVideos/grid1.mp4')

    texture = cv2.imread('/Users/wzdziechowski/Desktop/IA2Resources/Images/ITULogo.jpg')
    texture = cv2.pyrDown(texture)
    #find texture corners (start top left follow clockwise)
    mTex,nTex,t = texture.shape
    textureCorners = [(0.,0.),(float(mTex),0.),(float(mTex),float(nTex)),(0.,float(nTex))]

    running, imgOrig = cap.read()

    pattern_size = (9, 6)

    idx = [0,8,53,45]
    while(running):

        imgOrig = cv2.pyrDown(imgOrig)
        h, w, d = imgOrig.shape
        gray = cv2.cvtColor(imgOrig,cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, pattern_size)

        if found:
            term = ( cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1 )
            cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), term)
            cv2.drawChessboardCorners(imgOrig, pattern_size, corners, found)
            
        #    for t in idx:
        #        cv2.circle(imgOrig,(int(corners[t,0,0]),int(corners[t,0,1])),10,(255,t,t))

            #found image chessboard corners (start top left follow clockwise)
            chessCorners = [(corners[0,0,0],corners[0,0,1]),(corners[8,0,0],corners[8,0,1]),(corners[53,0,0],corners[53,0,1]),(corners[45,0,0],corners[45,0,1])]

            #Convert to openCV format
            ip1 = np.array([[x,y] for (x,y) in textureCorners])
            ip2 = np.array([[x,y] for (x,y) in chessCorners])
            
            

            #find homography
            H = SIGBTools.estimateHomography(ip1, ip2)

            #do the same as for the simple texture (add the images weighted)
            overlay = cv2.warpPerspective(texture, H,(w, h))
            imgOrig = cv2.addWeighted(imgOrig, 1, overlay ,0.5,0)

        cv2.imshow("win2",imgOrig)
        cv2.waitKey(1)
        running, imgOrig = cap.read()
    return None




def realisticTexturemap(scale,point,map):
    #H = np.load('H_G_M')
    print "Not implemented yet\n"*30


def showFloorTrackingData():
    #Load videodata
    fn = "GroundFloorData/sunclipds.avi"
    cap = cv2.VideoCapture(fn)
    
    #load Tracking data
    running, imgOrig = cap.read()
    dataFile = np.loadtxt('GroundFloorData/trackingdata.dat')
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
            cv2.imshow("boxes",imgOrig)
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
    fn = 'BookVideos/Seq3_scene.mp4'
    cap = cv2.VideoCapture(fn) 
    drawContours = True
    
    texture = cv2.imread('images/ITULogo.jpg')
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
                 #TO DO
                 
                 if(drawContours):                
                     for p in sqr:
                         cv2.circle(imgOrig,(int(p[0]),int(p[1])),3,(255,0,0)) 
                 
            
            if(drawContours and len(squares)>0):    
                cv2.drawContours( imgOrig, squares, -1, (0, 255, 0), 3 )

            cv2.circle(imgOrig,(100,100),10,(255,0,0))
            cv2.imshow("Detection",imgOrig)
            cv2.waitKey(1)


def DisplayTrace():

    #load the tracking data
    trackingData = loadtxt("GroundFloorData/trackingdata.dat")
#    print trackingData
    #gather into a matrix only the points representing the feet (P3 & P4)(P3x=point4, P3y=point5, P4x=point6, P5x=point7) using slicing
    feetTrackingData = trackingData[:,4:8]
#    print feetTrackingData
    #estimate the center of the feet (middle point of P3x and P4x, but only P4y) use slicing
    feetTrackingP3x = feetTrackingData [:,0:1]
    feetTrackingP4x = feetTrackingData [:,2:3]
    feetTrackingCx = (feetTrackingP3x+feetTrackingP4x)//2
    feetTrackingCy = feetTrackingData [:,3:4]
    m,n=feetTrackingData.shape
    feetTrackingC = feetTrackingData[:,0:2]
    for i in range(m):
        feetTrackingC[i][0]=feetTrackingCx[i][0]
        feetTrackingC[i][1]=feetTrackingCy[i][0]
#    print feetTrackingC

    #initialize images (I1=G, I2=M)
    cap = cv2.VideoCapture('GroundFloorData/SunClipDS.avi')
    running, I1 = cap.read()
    I2 = cv2.imread('Images/ITUMap.bmp')

    #Copy images
    drawImg = []
    drawImg.append(copy(cv2.cvtColor(I1,cv2.COLOR_BGR2RGB)))
    drawImg.append(copy(cv2.cvtColor(I2,cv2.COLOR_BGR2RGB)))

    imagePoints = []
    imagePoints=[[(89.,193.),(91.,142.),(213.,172.),(192.,135.)],[(330.,179.),(293.,169.),(337.,128.),(303.,128.)]]

    #Make figure
    fig = figure(1)
    for k in range(2):
        ax= subplot(1,2,k+1)
        ax.imshow(drawImg[k])
        ax.axis('image')
        title("Click 4 times in the  image")
        fig.canvas.draw()
        ax.hold('On')

        #Get mouse inputs
        #imagePoints.append(fig.ginput(4))

        #Draw selected points
        for p in imagePoints[k]:
            cv2.circle(drawImg[k],(int(p[0]),int(p[1])),1,(0,255,0),2)
        ax.imshow(drawImg[k])
        for (x,y) in imagePoints[k]:
            plot(x,y,'rx')
        fig.canvas.draw()

    #print imagePoints

    #Convert to openCV format
    ip1 = np.array([[x,y] for (x,y) in imagePoints[0]])
    ip2 = np.array([[x,y] for (x,y) in imagePoints[1]])

    #Calculate homography
    H,mask = cv2.findHomography(ip1, ip2)
    #print H
    np.save('Results/H_G_M',H)

    transformedPoints = list()

#    w = ip2[0]*ip1[0].I

    for p in feetTrackingC:
        currentPoint = np.matrix([p[0], p[1],1]).T
        transformedPoints.append((H*currentPoint).astype(int))

    for p in feetTrackingC:
        subplot(1,2,1)
#        plot(p[0],p[1],'rx')
        cv2.circle(I1,(int(p[0]),int(p[1])),1,(0,255,0),1)

    ax1=subplot(1,2,1)
    ax1.cla
    ax1.imshow(I1)
    draw() #update display: updates are usually deferred

    for p in transformedPoints:
        subplot(1,2,2)
#        plot(p[0],p[1],'rx')
        cv2.circle(I2,(int(p[0]),int(p[1])),2,(0,0,255),1)

    ax2=subplot(1,2,2)
    ax2.cla
    ax2.imshow(I2)
    draw() #update display: updates are usually deferred
    savefig('Results/pairedTracking.png')
    show()
    cv2.imwrite("Results/overviewTracking.jpg", I2)
    return transformedPoints

def DisplayTraceDynamic():
    #load the tracking data
    trackingData = loadtxt("GroundFloorData/trackingdata.dat")
    #    print trackingData
    #gather into a matrix only the points representing the feet (P3 & P4)(P3x=point4, P3y=point5, P4x=point6, P5x=point7) using slicing
    feetTrackingData = trackingData[:,4:8]
    #    print feetTrackingData
    #estimate the center of the feet (middle point of P3x and P4x, but only P4y) use slicing
    feetTrackingP3x = feetTrackingData [:,0:1]
    feetTrackingP4x = feetTrackingData [:,2:3]
    feetTrackingCx = (feetTrackingP3x+feetTrackingP4x)//2
    feetTrackingCy = feetTrackingData [:,3:4]
    m,n=feetTrackingData.shape
    feetTrackingC = feetTrackingData[:,0:2]
    for i in range(m):
        feetTrackingC[i][0]=feetTrackingCx[i][0]
        feetTrackingC[i][1]=feetTrackingCy[i][0]
    #    print feetTrackingC

    #initialize images (I1=G, I2=M)
    cap = cv2.VideoCapture('GroundFloorData/SunClipDS.avi')
    I2 = cv2.imread('Images/ITUMap.bmp')

    H = np.load('Results/H_G_M.npy')

    for p in feetTrackingC:
        running, I1 = cap.read()
        if(running):
            cv2.imshow("Sequence", I1)
            cv2.waitKey(1)
            currentPoint = matrix([p[0], p[1],1]).T
            hPoint = H * currentPoint
            cv2.circle(I2,(int(hPoint[0]),int(hPoint[1])),2,(0,255,0),1)
            cv2.imshow("Plan", I2)
            cv2.waitKey(1)

    return

def texturemapGroundFloor():
    I1 = cv2.imread('Images/ITULogo.jpg')
    cap = cv2.VideoCapture("GroundFloorData/sunclipds.avi")
    running, I2 = cap.read()
    H, points = SIGBTools.getHomographyFromMouse(I1,I2,-4)
    h, w, d = I2.shape
    overlay = cv2.warpPerspective(cv2.transpose(I1), H,(w, h))
    while(running):
        M = cv2.addWeighted(I2, 1.0, overlay, 0.5,0)
        cv2.imshow("Texture", M)
        cv2.waitKey(1)
        running, I2 = cap.read()
    return

def AugumentImages():
    #Loading callibration matrix
    K = np.load('Results/PMatrix.npy') 
    #setting pattern size
    pattern_size = (9,6)
    #loading calibration images
    L_CP = cv2.imread('Results/L_CP.jpg') #Frontal view
    #Getting cube points from cubePoints.py
    cubePoints = cube.cube_points([0,0,0.1],0.1)
    
    I1 = cv2.imread('Results/calibrationShoots1.jpg')
    I2 = cv2.imread('Results/calibrationShoots2.jpg')
    I3 = cv2.imread('Results/calibrationShoots3.jpg')
    I4 = cv2.imread('Results/calibrationShoots4.jpg')
    I5 = cv2.imread('Results/calibrationShoots5.jpg')
    
    Images = [I1, I2, I3, I4 ,I5]
    ImageH = [] #homographies from frontal view to H respectively
    
    #Getting chesscorners for frontal view
    Fgray = cv2.cvtColor(L_CP,cv2.COLOR_BGR2GRAY)
    Ffound, Fcorners = cv2.findChessboardCorners(Fgray, pattern_size)
    FchessCorners = [(Fcorners[0,0,0],Fcorners[0,0,1]),(Fcorners[8,0,0],Fcorners[8,0,1]),(Fcorners[45,0,0],Fcorners[45,0,1]),(Fcorners[53,0,0],Fcorners[53,0,1])]
    #To open CV format
    FchessCorners = np.array([[x,y] for (x,y) in FchessCorners])
    
    #Getting chesscorners for the rest of pics
    for I in Images:
        #converting to gray for better contrast
        gray = cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)
        #finding all corners on chessboard
        found, corners = cv2.findChessboardCorners(gray, pattern_size)
        #picking utmost corners
        IchessCorners = [(corners[0,0,0],corners[0,0,1]),(corners[8,0,0],corners[8,0,1]),(corners[45,0,0],corners[45,0,1]),(corners[53,0,0],corners[53,0,1])]
        #To openCV format
        IchessCorners = np.array([[x,y] for (x,y) in IchessCorners])
        H,mask = cv2.findHomography(FchessCorners, IchessCorners)
        ImageH.append(H)
        
    cam1 = SIGBTools.Camera(hstack((K,dot(K,array([[0],[0],[-1]])) )) )
    box_cam1 = cam1.project(SIGBTools.toHomogenious(cubePoints[:,:5]))
    
    cam2 = SIGBTools.Camera(dot(ImageH[3],cam1.P))
    A = dot(linalg.inv(K),cam2.P[:,:3])
    A = array([A[:,0],A[:,1],cross(A[:,0],A[:,1])]).T
    cam2.P[:,:3] = dot(K,A)
    box_cam2 = cam2.project(SIGBTools.toHomogenious(cubePoints))
    print box_cam2
    #figure()
    #imshow(I4) 
    #plot(box_cam2[0,:],box_cam2[1,:],linewidth=3)
    #show()
    p=box_cam2
    
    ''' Drawing the box manually '''
    #bottom
    cv2.line(I4, (int(p[0][1]), int(p[1][1])), (int(p[0][2]),int(p[1][2])),(255,255,0),2)
    cv2.line(I4, (int(p[0][2]), int(p[1][2])), (int(p[0][3]),int(p[1][3])),(255,255,0),2)
    cv2.line(I4, (int(p[0][3]), int(p[1][3])), (int(p[0][4]),int(p[1][4])),(255,255,0),2)
    cv2.line(I4, (int(p[0][1]), int(p[1][1])), (int(p[0][4]),int(p[1][4])),(255,255,0),2)
    
    #connecting lines
    cv2.line(I4, (int(p[0][4]), int(p[1][4])), (int(p[0][5]),int(p[1][5])),(255,255,0),2)
    cv2.line(I4, (int(p[0][1]), int(p[1][1])), (int(p[0][6]),int(p[1][6])),(255,255,0),2)
    cv2.line(I4, (int(p[0][2]), int(p[1][2])), (int(p[0][7]),int(p[1][7])),(255,255,0),2)
    cv2.line(I4, (int(p[0][3]), int(p[1][3])), (int(p[0][8]),int(p[1][8])),(255,255,0),2)
    
    #top
    cv2.line(I4, (int(p[0][5]), int(p[1][5])), (int(p[0][6]),int(p[1][6])),(255,255,0),2)
    cv2.line(I4, (int(p[0][6]), int(p[1][6])), (int(p[0][7]),int(p[1][7])),(255,255,0),2)
    cv2.line(I4, (int(p[0][7]), int(p[1][7])), (int(p[0][8]),int(p[1][8])),(255,255,0),2)
    cv2.line(I4, (int(p[0][8]), int(p[1][8])), (int(p[0][9]),int(p[1][9])),(255,255,0),2)
    
    cv2.imshow('Dupa',I4)
    cv2.waitKey(10000)
    
    return




def AugumentSequence():
    #Loading callibration matrix
    K = np.load('Results/PMatrix.npy') 
    #setting pattern size
    pattern_size = (9,6)
    #loading calibration images
    L_CP = cv2.imread('Results/L_CPSeq.jpg') #Frontal view
    #Getting cube points from cubePoints.py
    cubePoints = cube.cube_points([0,0,0.1],0.1)
    #Getting chesscorners for frontal view
    Fgray = cv2.cvtColor(L_CP,cv2.COLOR_BGR2GRAY)
    Ffound, Fcorners = cv2.findChessboardCorners(Fgray, pattern_size)
    FchessCorners = [(Fcorners[0,0,0],Fcorners[0,0,1]),(Fcorners[8,0,0],Fcorners[8,0,1]),(Fcorners[45,0,0],Fcorners[45,0,1]),(Fcorners[53,0,0],Fcorners[53,0,1])]
    #To open CV format
    FchessCorners = np.array([[x,y] for (x,y) in FchessCorners])
    #Getting chesscorners for the sequence images
    cap = cv2.VideoCapture(0)
    running, I = cap.read()
    #Get camera model for first view 
    cam1 = SIGBTools.Camera(hstack((K,dot(K,array([[0],[0],[-1]])) )) )
    
    #imSize = np.shape(I)
    #videoWriter = cv2.VideoWriter("sequence.mp4", cv.FOURCC('D','I','V','X'), 30,(imSize[1], imSize[0]),True)
    
    while(running):
        running, I = cap.read()
        #converting to gray for better contrast
        gray = cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, pattern_size)
        
        if (found):
        #picking utmost corners
            IchessCorners = [(corners[0,0,0],corners[0,0,1]),(corners[8,0,0],corners[8,0,1]),(corners[45,0,0],corners[45,0,1]),(corners[53,0,0],corners[53,0,1])]
            #To openCV format
            IchessCorners = np.array([[x,y] for (x,y) in IchessCorners])
            #Find homography between frontal image and current frame
            H,mask = cv2.findHomography(FchessCorners, IchessCorners)
            #Transofrm camera view
            camFrame = SIGBTools.Camera(dot(H,cam1.P))
            A = dot(linalg.inv(K),camFrame.P[:,:3])
            A = array([A[:,0],A[:,1],cross(A[:,0],A[:,1])]).T
            camFrame.P[:,:3] = dot(K,A)
            #Get cube projection points
            box_peojection = camFrame.project(SIGBTools.toHomogenious(cubePoints))
            #Draw  box
            p = box_peojection
            ''' Drawing the box manually '''
            #bottom
            cv2.line(I, (int(p[0][1]), int(p[1][1])), (int(p[0][2]),int(p[1][2])),(255,255,0),2)
            cv2.line(I, (int(p[0][2]), int(p[1][2])), (int(p[0][3]),int(p[1][3])),(255,255,0),2)
            cv2.line(I, (int(p[0][3]), int(p[1][3])), (int(p[0][4]),int(p[1][4])),(255,255,0),2)
            cv2.line(I, (int(p[0][1]), int(p[1][1])), (int(p[0][4]),int(p[1][4])),(255,255,0),2)
            
            #connecting lines
            cv2.line(I, (int(p[0][4]), int(p[1][4])), (int(p[0][5]),int(p[1][5])),(255,255,0),2)
            cv2.line(I, (int(p[0][1]), int(p[1][1])), (int(p[0][6]),int(p[1][6])),(255,255,0),2)
            cv2.line(I, (int(p[0][2]), int(p[1][2])), (int(p[0][7]),int(p[1][7])),(255,255,0),2)
            cv2.line(I, (int(p[0][3]), int(p[1][3])), (int(p[0][8]),int(p[1][8])),(255,255,0),2)
            
            #top
            cv2.line(I, (int(p[0][5]), int(p[1][5])), (int(p[0][6]),int(p[1][6])),(255,255,0),2)
            cv2.line(I, (int(p[0][6]), int(p[1][6])), (int(p[0][7]),int(p[1][7])),(255,255,0),2)
            cv2.line(I, (int(p[0][7]), int(p[1][7])), (int(p[0][8]),int(p[1][8])),(255,255,0),2)
            cv2.line(I, (int(p[0][8]), int(p[1][8])), (int(p[0][9]),int(p[1][9])),(255,255,0),2)
            
            cv2.imshow('Augumentation',I)
            cv2.waitKey(1)
            
    return

    



# Main
#showImageandPlot(4)
#DisplayTrace()
#DisplayTraceDynamic()
#texturemapGroundFloor()

#showFloorTrackingData()
#simpleTextureMap()
#realisticTexturemap(0,0,0)
#texturemapGridSequence()
#calibrationExample()
#AugumentImages()
AugumentSequence()
