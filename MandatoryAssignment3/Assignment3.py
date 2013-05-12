'''
Created on Apr 11, 2013
@author: Diako Mardanbegi (dima@itu.dk)
'''
from numpy import *
import numpy as np
from pylab import *
from scipy import linalg
import cv2
import cv2.cv as cv
from SIGBTools import *
import math

def DrawLines(img, points):
    for i in range(1, 17):                
         x1 = points[0, i - 1]
         y1 = points[1, i - 1]
         x2 = points[0, i]
         y2 = points[1, i]
         cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0),5) 
    return img

def getCubePoints(center, size,chessSquare_size):

    """ Creates a list of points for plotting a cube with plot.
    (the first 5 points are the bottom square, some sides repeated). """
    points = []
    
    """
    1    2
        5    6
    3    4
        7    8 (bottom)
    
    """

    #bottom
    points.append([center[0]-size, center[1]-size, center[2]-2*size])#(0)5
    points.append([center[0]-size, center[1]+size, center[2]-2*size])#(1)7
    points.append([center[0]+size, center[1]+size, center[2]-2*size])#(2)8
    points.append([center[0]+size, center[1]-size, center[2]-2*size])#(3)6
    points.append([center[0]-size, center[1]-size, center[2]-2*size]) #same as first to close plot
    
    #top
    points.append([center[0]-size,center[1]-size,center[2]])#(5)1
    points.append([center[0]-size,center[1]+size,center[2]])#(6)3
    points.append([center[0]+size,center[1]+size,center[2]])#(7)4
    points.append([center[0]+size,center[1]-size,center[2]])#(8)2
    points.append([center[0]-size,center[1]-size,center[2]]) #same as first to close plot
    
    #vertical sides
    points.append([center[0]-size,center[1]-size,center[2]])
    points.append([center[0]-size,center[1]+size,center[2]])
    points.append([center[0]-size,center[1]+size,center[2]-2*size])
    points.append([center[0]+size,center[1]+size,center[2]-2*size])
    points.append([center[0]+size,center[1]+size,center[2]])
    points.append([center[0]+size,center[1]-size,center[2]])
    points.append([center[0]+size,center[1]-size,center[2]-2*size])
    points=dot(points,chessSquare_size)
    return array(points).T


def update(img):
    image=copy(img)
    #w,h,z = shape(image)

    if Undistorting:  #Use previous stored camera matrix and distortion coefficient to undistort the image
        ''' <004> Here Undistoret the image'''
        image=cv2.undistort(image,cameraMatrix,distortionCoefficient)
    
    if ProcessFrame:

        ''' <005> Here Find the Chess pattern in the current frame'''
        patternFound,corners = cv2.findChessboardCorners(image,(9,6))

        if patternFound ==True:
            
            ''' <006> Here Define the cameraMatrix P=K[R|t] of the current frame'''
            if debug:
                P2=P2_M1(calibrationPoints,corners,cam1)
                #print "M1"
                #print P2
            else:
                P2=P2_M2(corners,cameraMatrix,distortionCoefficient)
                #print "M2"
                #print P2
            cam2=Camera(P2)

            if ShowText:
                ''' <011> Here show the distance between the camera origin and the world origin in the image'''
                cv2.putText(image,str("distance:"+str(getDistance(cam2))),(20,10),cv2.FONT_HERSHEY_PLAIN,1, (0, 0, 0))
                cv2.putText(image,str("frame:" + str(frameNumber)), (20,20),cv2.FONT_HERSHEY_PLAIN,1, (0, 0, 0))#Draw the text

            ''' <008> Here Draw the world coordinate system in the image'''
            image = drawWorldCoordinateSystem(image,cam2)

            TopFaceCornerNormals,RightFaceCornerNormals,LeftFaceCornerNormals,UpFaceCornerNormals,DownFaceCornerNormals=CalculateFaceCornerNormals(TopFace,RightFace,LeftFace,UpFace,DownFace)

            if TextureMap:
                ''' <010> Here Do the texture mapping and draw the texture on the faces of the cube'''
                ''' <012> Here Remove the hidden faces'''
                ''' <013> Here Remove the hidden faces'''
                #determine which face needs to be projected
                (show_T,show_R,show_L,show_U,show_D) = normalVectors(image,cam2)
                #texture the faces
                image = textureMap(image,cam2,show_T,show_R,show_L,show_U,show_D)
                #re-draw normal vectors
                normalVectors(image,cam2)
                #apply shading to visible faces
                if show_T: image=ShadeFace(image,TopFace,TopFaceCornerNormals,cam2)
                if show_R: image=ShadeFace(image,RightFace,RightFaceCornerNormals,cam2)
                if show_L: image=ShadeFace(image,LeftFace,LeftFaceCornerNormals,cam2)
                if show_U: image=ShadeFace(image,UpFace,UpFaceCornerNormals,cam2)
                if show_D: image=ShadeFace(image,DownFace,DownFaceCornerNormals,cam2)

            if ProjectPattern:
                ''' <007> Here Test the camera matrix of the current view by projecting the pattern points''' 
                image=checkAnyView(image,cam2)

            if WireFrame:                      
                ''' <009> Here Project the box into the current camera image and draw the box edges'''
                image = projectTheBox(image, cam2)

    cv2.imshow('Web cam', image)  
    global result
    result=copy(image)

def getImageSequence(capture, fastForward):
    '''Load the video sequence (fileName) and proceeds, fastForward number of frames.'''
    global frameNumber
   
    for t in range(fastForward):
        isSequenceOK, originalImage = capture.read()  # Get the first frames
        frameNumber = frameNumber+1
    return originalImage, isSequenceOK


def printUsage():
    print "Q or ESC: Stop"
    print "SPACE: Pause"     
    print "p: turning the processing on/off "  
    print 'u: undistorting the image'
    print 'i: show info'
    print 't: texture map'
    print 'g: project the pattern using the camera matrix (test)'
    print 's: save frame'
    print 'x: do something!'
    
   
def run(speed): 
    
    '''MAIN Method to load the image sequence and handle user inputs'''   

    #--------------------------------video
    capture = cv2.VideoCapture("/Users/wzdziechowski/Desktop/Pattern.mov")
    #--------------------------------camera
    #capture = cv2.VideoCapture(0)

    image, isSequenceOK = getImageSequence(capture,speed)       

    if isSequenceOK:
        update(image)
        printUsage()

    while isSequenceOK:
        OriginalImage=copy(image)

        inputKey = cv2.waitKey(1)
        
        if inputKey == 32:#  stop by SPACE key
            update(OriginalImage)
            if speed==0:     
                speed = tempSpeed
            else:
                tempSpeed=speed
                speed = 0
            
        if (inputKey == 27) or (inputKey == ord('q')):#  break by ECS key
            break    
                
        if inputKey == ord('p') or inputKey == ord('P'):
            global ProcessFrame
            if ProcessFrame:     
                ProcessFrame = False
                
            else:
                ProcessFrame = True
            update(OriginalImage)
            
        if inputKey == ord('u') or inputKey == ord('U'):
            global Undistorting
            if Undistorting:     
                Undistorting = False
            else:
                Undistorting = True
            update(OriginalImage)     
        if inputKey == ord('w') or inputKey == ord('W'):
            global WireFrame
            if WireFrame:     
                WireFrame = False
                
            else:
                WireFrame = True
            update(OriginalImage)

        if inputKey == ord('i') or inputKey == ord('I'):
            global ShowText
            if ShowText:     
                ShowText = False
                
            else:
                ShowText = True
            update(OriginalImage)
            
        if inputKey == ord('t') or inputKey == ord('T'):
            global TextureMap
            if TextureMap:     
                TextureMap = False
                
            else:
                TextureMap = True
            update(OriginalImage)
            
        if inputKey == ord('g') or inputKey == ord('G'):
            global ProjectPattern
            if ProjectPattern:     
                ProjectPattern = False
                
            else:
                ProjectPattern = True
            update(OriginalImage)   
             
        if inputKey == ord('x') or inputKey == ord('X'):
            global debug
            if debug:     
                debug = False
            else:
                debug = True
            update(OriginalImage)   
            
                
        if inputKey == ord('s') or inputKey == ord('S'):
            name='Saved Images/Frame_' + str(frameNumber)+'.png' 
            cv2.imwrite(name,result)
           
        if (speed>0):
            update(image)
            image, isSequenceOK = getImageSequence(capture,speed)          

#---Global variables
global cameraMatrix #intrinsic cameraMatrix (K or A) ... not full matrix (K[R|t] or A[R|t])
global distortionCoefficient
global homographyPoints
global calibrationPoints
global calibrationCamera
global chessSquare_size
global box
global TopFace
global RightFace
global LeftFace
global UpFace
global DownFace
global cam1
global cam2
    
ProcessFrame=False
Undistorting=False   
WireFrame=False
ShowText=True
TextureMap=True
ProjectPattern=False
debug=True

frameNumber=0


chessSquare_size=2
            
box = getCubePoints([4, 2.5, 0], 1,chessSquare_size)            

i = array([ [0,0,0,0],[1,1,1,1] ,[2,2,2,2]  ])  # indices for the first dim
j = array([ [0,3,2,1],[0,3,2,1] ,[0,3,2,1]  ])  # indices for the second dim            
TopFace = box[i,j]

i = array([ [0,0,0,0],[1,1,1,1] ,[2,2,2,2]  ])  # indices for the first dim 
j = array([ [3,8,7,2],[3,8,7,2] ,[3,8,7,2]  ])  # indices for the second dim
RightFace = box[i,j]

i = array([ [0,0,0,0],[1,1,1,1] ,[2,2,2,2]  ])  # indices for the first dim 
j = array([ [5,0,1,6],[5,0,1,6] ,[5,0,1,6]  ])  # indices for the second dim            
LeftFace = box[i,j]

i = array([ [0,0,0,0],[1,1,1,1] ,[2,2,2,2]  ])  # indices for the first dim 
j = array([ [5,8,3,0], [5,8,3,0] , [5,8,3,0] ])  # indices for the second dim            
UpFace = box[i,j]

i = array([ [0,0,0,0],[1,1,1,1] ,[2,2,2,2]  ])  # indices for the first dim 
j = array([ [1,2,7,6], [1,2,7,6], [1,2,7,6] ])  # indices for the second dim            
DownFace = box[i,j]


'''----------------------------------------'''
def calculateFullCameraMatrix(cameraMatrix,rotationVectors,translationVectors):
    #convert rotation vector to matrix
    rotationMatrix=cv2.Rodrigues(rotationVectors)
    #add the translation to the last column of the rotation matrix
    extrinsic = np.hstack((rotationMatrix[0],translationVectors)) #extrinsic camera matrix for 1st view ([R|t])
    #remember to convert the 'np.arrays' to 'np.matrices' for the multiplication
    #multiply the intrinsic and extrinsic matrices
    P = np.matrix(cameraMatrix) * np.matrix(extrinsic) #full camera matrix for 1st view (K[R|t] or A[R|t])
    return P

def checkFirstView(cam1):
    first_view = cv2.imread('01.png')

    corners = np.matrix([[0,0,0,1],[0,10,0,1],[16,0,0,1],[16,10,0,1]])

    for i in range(len(corners)):
        projected_corner=cam1.project(corners[i].T)
        cv2.circle(first_view,(int(projected_corner[0]),int(projected_corner[1])),5,(0,255,255))

    cv2.imshow('checking',first_view)
    cv2.waitKey(4000)
    return None

def checkAnyView(image,cam2):
    corners = np.matrix([[0,0,0,1],[0,10,0,1],[16,0,0,1],[16,10,0,1]])

    for i in range(len(corners)):
        projected_corner=cam2.project(corners[i].T)
        projected_corner=normalizeHomogenious(projected_corner)
        cv2.circle(image,(projected_corner[0],projected_corner[1]),5,(0,255,255))

    cv2.imshow('checking',image)
    return image

def P2_M1(calibrationPoints,corners,cam1):
    # first method P2_Method1 --- use data from first and current view
    # get utmost corners of first view
    corners1 = [(calibrationPoints[0][0,0],calibrationPoints[0][0,1]),(calibrationPoints[0][8,0],calibrationPoints[0][8,1]),(calibrationPoints[0][45,0],calibrationPoints[0][45,1]),(calibrationPoints[0][53,0],calibrationPoints[0][53,1])]
    # get utmost corners of current frame
    corners2 = [(corners[0,0,0],corners[0,0,1]),(corners[8,0,0],corners[8,0,1]),(corners[45,0,0],corners[45,0,1]),(corners[53,0,0],corners[53,0,1])]
    # to openCV format
    corners1 = np.array([[x,y] for (x,y) in corners1])
    corners2 = np.array([[x,y] for (x,y) in corners2])
    # find homography between first and current view
    H,mask = cv2.findHomography(corners1, corners2)

    #correct 3rd column rotation of camera1 r3=r1 cross r2
    cam1=Camera(P1)
    (K,R,T)=cam1.factor()
    cam2 = Camera(dot(H,P1))
    A = dot(linalg.inv(K),np.array(cam2.P[:,:3]))
    A = array([A[:,0],A[:,1],cross(A[:,0],A[:,1])]).T
    cam2.P[:,:3] = dot(K,A)

    # compute second camera matrix from cam1 and H
    cam2 = Camera(dot(H,cam1.P))

    return cam2.P

def P2_M2(corners,cameraMatrix,distortionCoefficient):
    # second method P2_Method2 --- use data from current view: object 3D to image 2D
    pattern_points = np.zeros((np.prod((9,6)), 3), np.float32 )
    pattern_points[:,:2] = np.indices((9,6)).T.reshape(-1, 2)
    pattern_points *= chessSquare_size
    obj_points = [pattern_points]
    obj_points.append(pattern_points)
    obj_points = np.array(obj_points,np.float64).T
    obj_points=obj_points[:,:,0].T

    found,rvecs_new,tvecs_new =cv2.solvePnP(obj_points,corners,cameraMatrix,distortionCoefficient)#obj_points (3xN)/(Nx3) or (1xN)/(Nx1) ; img_points (2xN)/(Nx2) or (1xN)/(Nx1)

    P2_Method2 = calculateFullCameraMatrix(cameraMatrix,rvecs_new,tvecs_new)
    return P2_Method2

def drawWorldCoordinateSystem(image, cam2):
    O =[[0],[0],[0],[1]];    OX=[[2],[0],[0],[1]];    OY=[[0],[2],[0],[1]];    OZ=[[0],[0],[2],[1]]
    Op = cam2.project(O);     Op = Op[:2]
    OXp= cam2.project(OX);    OXp = OXp[:2]
    OYp= cam2.project(OY);    OYp = OYp[:2]
    OZp= cam2.project(OZ);    OZp = OZp[:2]
    cv2.circle(image,(int(OXp[0]),int(OXp[1])),2,(255,0,0))
    cv2.line(image,(int(Op[0]),int(Op[1])),(int(OXp[0]),int(OXp[1])),(0,255,255))
    cv2.circle(image,(int(OYp[0]),int(OYp[1])),2,(0,255,0))
    cv2.line(image,(int(Op[0]),int(Op[1])),(int(OYp[0]),int(OYp[1])),(0,255,255))
    cv2.circle(image,(int(OZp[0]),int(OZp[1])),2,(0,0,255))
    cv2.line(image,(int(Op[0]),int(Op[1])),(int(OZp[0]),int(OZp[1])),(0,255,255))
    return image

def projectTheBox(image, cam2):
    box_projection = cam2.project(toHomogenious(box))
    image = DrawLines(image, box_projection)
    return image

def textureMap(image,cam2,show_T,show_R,show_L,show_U,show_D):
    (y,x,z) = shape(image)
    if show_R: image=textureMapFace(image,RightFace,'Right.jpg',cam2,x,y)
    if show_L: image=textureMapFace(image,LeftFace,'Left.jpg',cam2,x,y)
    if show_U: image=textureMapFace(image,UpFace,'Up.jpg',cam2,x,y)
    if show_D: image=textureMapFace(image,DownFace,'Down.jpg',cam2,x,y)
    if show_T: image=textureMapFace(image,TopFace,'Top.jpg',cam2,x,y)
    return image

def textureMapFace(image,face,faceJPG,cam2,x,y):
    texture_image = cv2.imread('Images/'+faceJPG)
    h,w,d = shape(texture_image)
    # find the corners of the texture
    texture_corners=np.array([[0.,0.],[float(w),0.],[float(w),float(h)],[0.,float(h)]])
    # find the corners of the projective face using the camera matrix

    # split to individual points, homogenize them, project them, un-homogenize the projected points
    projected_corners=[]
    for i in range(4):
        # split and homogenize
        corner = hstack((face.T[i],1))
        # project
        projected_corner = cam2.project(np.matrix(corner).T)
        # un-homogenize
        projected_corners.append(projected_corner[:2])
    projected_corners=np.array(projected_corners)
    H=estimateHomography(texture_corners,projected_corners)

    texture = cv2.warpPerspective(texture_image,H,(x,y)) # black image with projected face

    white_image = cv2.imread('Images/white.jpg')

    Mask = cv2.warpPerspective(white_image,H,(x,y)) # black image with white face
    Mask=cv2.bitwise_not(Mask) # white image with black face

    I1=cv2.bitwise_and(Mask,image) # image background with black face
    image=cv2.bitwise_or(I1,texture) # image background with texture image face

    #overlay = cv2.warpPerspective(texture_image,H,(x,y))
    #image=cv2.addWeighted(image,0.5,overlay,0.5,0)

    return image

def getDistance(cam2):
    (K,R,T)=cam2.factor()
    return str(T[2,0])

def normalVectors(image,cam2):
    show_T=normalVectorFace(image,cam2,TopFace,'t')
    show_R=normalVectorFace(image,cam2,RightFace,'r')
    show_L=normalVectorFace(image,cam2,LeftFace,'l')
    show_U=normalVectorFace(image,cam2,UpFace,'u')
    show_D=normalVectorFace(image,cam2,DownFace,'d')
    return show_T,show_R,show_L,show_U,show_D

def normalVectorFace(image,cam2,face,f):
    #find the center by choosing diagonal points of the face and averaging their coordinates
    #face_center_T=[(x1+x2)/2,(y1+y2)/2,(z1+z2)/2]
    tf = face.T
    face_center=np.array([(tf[0][0]+tf[2][0])/2,(tf[0][1]+tf[2][1])/2,(tf[0][2]+tf[2][2])/2])

    #try 3
    # N = [fx(x0,y0),fy(x1,y1),fz(x2,y2)]
    if f=='t':
        v_n=np.array([0.,0.,-1.])
    elif f=='r':
        v_n=np.array([1.,0.,0.])
    elif f=='l':
        v_n=np.array([-1.,0.,0.])
    elif f=='u':
        v_n=np.array([0.,-1.,0.])
    elif f=='d':
        v_n=np.array([0.,1.,0.])
    else:
        v_n=np.array([0.,0.,0.])

    #try 4 - cross product
    #face=face.T
    #v1=face[0]-face[1]
    #v2=face[2]-face[1]
    #v_n=cross(v1,v2)

    #SIGBTools
    #v_n=GetFaceNormal(face)

    # choose a point on the normal vector
    d_n=np.array(face_center).T+np.array(v_n).T

    # back-face culling
    # estimate the angle between the normal vector and the line passing through the camera origin and the face center
    # normal vector (n) (normal point - face center)
    # vector passing through camera center and face center (camera center - face center)
    camera_center=np.array([cam2.center()[0,0],cam2.center()[1,0],cam2.center()[2,0]]).T
    v_cc_fc=camera_center-face_center

    # calculate their dot product
    dot_n_cc_fc=dot(v_n,v_cc_fc)
    # calculate magnitudes
    m_v_n=math.sqrt(v_n[0]**2+v_n[1]**2+v_n[2]**2)
    m_v_cc_fc=math.sqrt(v_cc_fc[0]**2+v_cc_fc[1]**2+v_cc_fc[2]**2)
    # find the cos of the angle --- cos(angle) = (dot product between the vectors)/(their magnitude multiplication)
    cos_angle=dot_n_cc_fc/(m_v_n*m_v_cc_fc)
    # find the angle --- inverse cos(angle) = arccos(resulting cos)
    angle=arccos(cos_angle) * 180.0/ pi # radians to degrees

    show=False
    # show the normal if the face is visible
    if angle<88:
        show=True

        #homogenize the 2 points
        face_center=hstack((face_center,1)); face_center=[[face_center[0]],[face_center[1]],[face_center[2]],[face_center[3]]]
        d_n=hstack((d_n,1)); d_n=[[d_n[0]],[d_n[1]],[d_n[2]],[d_n[3]]]

        # project the 2 points
        face_center_p=cam2.project(face_center)
        d_n_p=cam2.project(d_n)

        # draw a line
        cv2.circle(image,(int(face_center_p[0]),int(face_center_p[1])),2,(255,0,0))
        cv2.line(image,(int(face_center_p[0]),int(face_center_p[1])),(int(d_n_p[0]),int(d_n_p[1])),(0,255,255))

    return show

def ShadeFace(image,points,faceCorner_Normals, camera):
    global shadeRes
    shadeRes=10
    videoHeight, videoWidth, vd = array(image).shape

    #find image face coordinates
    points_Proj=camera.project(toHomogenious(points))
    points_Proj1 = np.array([[int(points_Proj[0,0]),int(points_Proj[1,0])],[int(points_Proj[0,1]),int(points_Proj[1,1])],[int(points_Proj[0,2]),int(points_Proj[1,2])],[int(points_Proj[0,3]),int(points_Proj[1,3])]])

    #corners of a square
    square = np.array([[0, 0], [shadeRes-1, 0], [shadeRes-1, shadeRes-1], [0, shadeRes-1]])

    #find the homography (square -> image face coordinates)
    H = estimateHomography(square, points_Proj1)

    #find the combined Phong illumination(shading) for each channel of the square
    Mr0,Mg0,Mb0=CalculateShadeMatrix_Interpolated(image,shadeRes,points,faceCorner_Normals, camera)
    # HINT
    # type(Mr0): <type 'numpy.ndarray'>
    # Mr0.shape: (shadeRes, shadeRes)

    #warp the illumination(shading) from the 2D square to the 2D image
    Mr = cv2.warpPerspective(Mr0, H, (videoWidth, videoHeight),flags=cv2.INTER_LINEAR)
    Mg = cv2.warpPerspective(Mg0, H, (videoWidth, videoHeight),flags=cv2.INTER_LINEAR)
    Mb = cv2.warpPerspective(Mb0, H, (videoWidth, videoHeight),flags=cv2.INTER_LINEAR)

    #split the channels (image intensity for each pixel)
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    [r,g,b]=cv2.split(image)

    #create a white texture mapped to the image coordinates of the face
    whiteMask = np.copy(r)
    whiteMask[:,:]=[0]
    points_Proj2=[]
    points_Proj2.append([int(points_Proj[0,0]),int(points_Proj[1,0])])
    points_Proj2.append([int(points_Proj[0,1]),int(points_Proj[1,1])])
    points_Proj2.append([int(points_Proj[0,2]),int(points_Proj[1,2])])
    points_Proj2.append([int(points_Proj[0,3]),int(points_Proj[1,3])])
    cv2.fillConvexPoly(whiteMask,array(points_Proj2),(255,255,255))

    #apply the Phong illumination (shading) for each channel to the white texture mapped image
    r[nonzero(whiteMask>0)]=map(lambda x: max(min(x,255),0),r[nonzero(whiteMask>0)]*Mr[nonzero(whiteMask>0)])
    g[nonzero(whiteMask>0)]=map(lambda x: max(min(x,255),0),g[nonzero(whiteMask>0)]*Mg[nonzero(whiteMask>0)])
    b[nonzero(whiteMask>0)]=map(lambda x: max(min(x,255),0),b[nonzero(whiteMask>0)]*Mb[nonzero(whiteMask>0)])

    #merge the channels
    image=cv2.merge((r,g,b))
    image=cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image

#flat shading using only face center
def CalculateShadeMatrix(image,shadeRes,points,faceCorner_Normals,camera):
    #Ambient light IA=[IaR,IaG,IaB]
    IA = np.matrix([5.0, 5.0, 5.0]).T

    #Point light IA=[IpR,IpG,IpB]
    IP = np.matrix([5.0, 5.0, 5.0]).T

    #Light Source Attenuation
    fatt = 1

    #Material properties: e.g., Ka=[kaR; kaG; kaB]
    ka=np.matrix([0.2, 0.2, 0.2]).T
    kd= np.matrix([0.3, 0.3, 0.3]).T
    ks=np.matrix([0.7, 0.7, 0.7]).T

    #object/face sharpness - shiny surface
    alpha = 100

    #need just 1 normal - the one to the face
    n=GetFaceNormal(points)

    #need face center
    vertices=np.array(points).T
    face_center=(vertices[0]+vertices[2])/2

    #need direction to viewer/camera (v) -> (1x3) array for the face center
    v=normalizeVecotr(np.array(camera.center().T)[0]-face_center)

    #need direction to the light source (l) - find v first -> (1x3) array for the face center
    #we assume that the light is coming from the center of the camera, therefore l==v
    l=v

    #need direction to the perfect reflection (r) - find l first -> (3x4) matrix for the 4 vertices
    #R = (2 x N x (N . L)) - L
    r=2*n*dot(n,l)-l

    #find n dot l
    ndl=dot(n,l)

    #find r dot v
    rdv=dot(r,v)

    #find the Phong illumination value
    #Iambient (x) = Ia ka(x)     #no attenuation
    #Idiffuse (x) = Il(x) kd(x) max(n(x) . l(x), 0)     #attenuation
    #Iglossy (x) = Is ks(r . v)^alpha     #attenuation
    #Phong = Iambient + Idiffuse + Iglossy

    #attenuation
    #I=kaIa+SUM(fatt(kd(N . L) Id +ks (N . H)^n Is))

    Iambient= ka[0,0] * IA[0,0]                         #0.2 * 5 = 1
    Idiffuse= kd[0,0] * IP[0,0] * max(ndl,0)            #0.3 * 5 * ndl<0:1> = <0:1.5>
    Iglossy = ks[0,0] * IP[0,0] * max(rdv**alpha,0)     #0.7 * 5 * rdv<0:1> ^ 100 = 3.5 *0.0... ~ 0
    I=Iambient+fatt*(Idiffuse+Iglossy)

    #create a shading matrix with the size of the square used for the homography and intensity equal to the Phong illumination
    Mr0 = np.zeros(shape=(shadeRes,shadeRes))
    Mr0[:,:]=[I]

    #repeat for other channels
    Mg0=copy(Mr0)
    Mb0=copy(Mr0)

    return Mr0,Mg0,Mb0

#flat shading using average of vertices - slow and misleading
def CalculateShadeMatrix_Average(image,shadeRes,points,faceCorner_Normals,camera):
    #Ambient light IA=[IaR,IaG,IaB]
    IA = np.matrix([5.0, 5.0, 5.0]).T

    #Point light IA=[IpR,IpG,IpB]
    IP = np.matrix([5.0, 5.0, 5.0]).T

    #Light Source Attenuation
    fatt = 1

    #Material properties: e.g., Ka=[kaR; kaG; kaB]
    ka=np.matrix([0.2, 0.2, 0.2]).T
    kd= np.matrix([0.3, 0.3, 0.3]).T
    ks=np.matrix([0.7, 0.7, 0.7]).T

    #object/face sharpness - shiny surface
    alpha = 100

    #have normal vectors to the points (n) -> (3x4) matrix for the 4 vertices
    n=np.matrix(faceCorner_Normals)

    #need direction to viewer/camera (v) -> (3x4) matrix for the 4 vertices
    vertices=np.array(points).T
    v=np.matrix(normalizeVecotr((camera.center()-vertices[0:1].T).T)).T
    for i in range(len(vertices)-1):
        v=vstack((v,np.matrix(normalizeVecotr((camera.center()-vertices[i+1:i+2].T).T)).T))
    v=v.T

    #need direction to the light source (l) - find v first -> (3x4) matrix for the 4 vertices
    #we assume that the light is coming from the center of the camera, therefore l==v
    l=v

    #need direction to the perfect reflection (r) - find l first -> (3x4) matrix for the 4 vertices
    #R = (2 x N x (N . L)) - L
    r=(2*n.T[0:1]*dot(n.T[0:1],l.T[0:1].T)[0,0])-l.T[0:1]
    for i in range(len(vertices)-1):
        r=vstack((r,(2*n.T[i+1:i+2]*dot(n.T[i+1:i+2],l.T[i+1:i+2].T)[0,0])-l.T[i+1:i+2]))
    r=r.T

    #find n dot l for each vertex -> (1x4) matrix for the 4 vertices
    ndl=dot(n.T[0:1],l.T[0:1].T)[0,0]
    for i in range(len(vertices)-1):
        ndl=vstack((ndl,dot(n.T[i+1:i+2],l.T[i+1:i+2].T)[0,0]))
    ndl=ndl.T

    #find r dot v for each vertex -> (1x4) matrix for the 4 vertices
    rdv=dot(r.T[0:1],v.T[0:1].T)[0,0]
    for i in range(len(vertices)-1):
        rdv=vstack((rdv,dot(r.T[i+1:i+2],v.T[i+1:i+2].T)[0,0]))
    rdv=rdv.T

    #find the 4 individual Phong vertex values and average them
    #Iambient (x) = Ia ka(x)     #no attenuation
    #Idiffuse (x) = Il(x) kd(x) max(n(x) . l(x), 0)     #attenuation
    #Iglossy (x) = Is ks(r . v)^alpha     #attenuation
    #Phong = Iambient + Idiffuse + Iglossy

    #attenuation
    #I=kaIa+SUM(fatt(kd(N . L) Id +ks (N . H)^n Is))

    I=[]
    for i in range(len(vertices)):
        #find 1 vertex Phong illumination for 1 channel (red)
        Iambient1= ka[0,0] * IA[0,0]                           #0.2 * 5 = 1
        Idiffuse1= kd[0,0] * IP[0,0] * max(ndl[0,0],0)         #0.3 * 5 * ndl<0:1> = <0:1.5>
        Iglossy1 = ks[0,0] * IP[0,0] * max(rdv[0,0]**alpha,0)  #0.7 * 5 * rdv<0:1> ^ 100 = 3.5 *0.0... ~ 0
        I1=Iambient1+fatt*(Idiffuse1+Iglossy1)
        I.append(I1)

    #average them for flat shading
    av=I[0]
    for i in range(len(vertices)-1):
        av+=I[i+1]
    av=av/len(vertices)

    #create a shading matrix with the size of the face and intensity equal to the averaged shading
    Mr0 = np.zeros(shape=(shadeRes,shadeRes))
    Mr0[:,:]=[av]

    #repeat for other channels
    Mg0=copy(Mr0)
    Mb0=copy(Mr0)

    return Mr0,Mg0,Mb0

#interpolated shading
def CalculateShadeMatrix_Interpolated(image,shadeRes,points,faceCorner_Normals,camera):
    #Ambient light IA=[IaR,IaG,IaB]
    IA = np.matrix([5.0, 5.0, 5.0]).T

    #Point light IA=[IpR,IpG,IpB]
    IP = np.matrix([5.0, 5.0, 5.0]).T

    #Light Source Attenuation
    fatt = 1

    #Material properties: e.g., Ka=[kaR; kaG; kaB]
    ka=np.matrix([0.2, 0.2, 0.2]).T
    kd= np.matrix([0.3, 0.3, 0.3]).T
    ks=np.matrix([0.7, 0.7, 0.7]).T

    #object/face sharpness - shiny surface
    alpha = 100

    #create a shading matrix with the size of the square
    Mr0 = np.zeros(shape=(shadeRes,shadeRes))

    for i in range(shadeRes):
        for j in range(shadeRes):
            #interpolate normals
            (X,Y,Z) = BilinearInterpo(shadeRes,i,j,faceCorner_Normals,True)
            n=np.matrix([[X,Y,Z]])
            #interpolate vector from the point to the view/light source
            (I,J,K) = BilinearInterpo(shadeRes,i,j,points,False)
            v=np.matrix(normalizeVecotr(camera.center()-np.matrix([[I],[J],[K]])))
            l=v
            #calculate perfect reflection vector    R = (2 x N x (N . L)) - L
            r=(2*n*dot(n,l.T)[0,0])-l

            #calculate Phong illumination value
            Mr0[j,i]=ka[0,0] * IA[0,0]+fatt*(kd[0,0] * IP[0,0] * max(dot(n,l.T)[0,0],0)+ks[0,0] * IP[0,0] * max(dot(r,v.T)[0,0]**alpha,0))

    #repeat for other channels
    Mg0=copy(Mr0)
    Mb0=copy(Mr0)

    return Mr0,Mg0,Mb0


'''----------------------------------------'''



''' <000> Here Call the cameraCalibrate2 from the SIGBTools to calibrate the camera and saving the data'''
#cameraCalibrate2(fileName=0)
#RecordVideoFromCamera()
''' <001> Here Load the numpy data files saved by the cameraCalibrate2'''
cameraMatrix          =np.load('numpyData/camera_matrix.npy')
chessSquare_size      =np.load('numpyData/chessSquare_size.npy')
distortionCoefficient =np.load('numpyData/distortionCoefficient.npy')
calibrationPoints     =np.load('numpyData/img_points.npy')
img_points_first      =np.load('numpyData/img_points_first.npy')
obj_points            =np.load('numpyData/obj_points.npy')
rotationVectors       =np.load('numpyData/rotatioVectors.npy')
translationVectors    =np.load('numpyData/translationVectors.npy')
''' <002> Here Define the camera matrix of the first view image (01.png) recorded by the cameraCalibrate2'''
P1=calculateFullCameraMatrix(cameraMatrix,rotationVectors[0],translationVectors[0])
cam1=Camera(P1)
''' <003> Here Load the first view image (01.png) and find the chess pattern and store the 4 corners of the pattern needed for homography estimation'''
#checkFirstView(cam1)

run(1)