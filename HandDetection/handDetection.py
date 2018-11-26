'''
El problema de la deteccion de la mano se divide en 3 partes :
    1. Remover el fondo 
    2. Deteccion de movimiento y binarizacion
    3. Deteccion del contorno
'''

import cv2
import numpy as np
import imutils
import math

#global variables
bg = None

def run_avg(image, aWeight):
    global bg
    # initialize the background in the first frame
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, aWeight)

def segment(image, threshold=25):
    global bg
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff,
                                threshold,
                                255,
                                cv2.THRESH_BINARY)[1]

    # get the contours in the thresholded image
    (_, cnts, _) = cv2.findContours(thresholded.copy(),
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)

if __name__ == "__main__":
    # initialize weight for running average
    aWeight = 0.5

    # get the reference to the webcam
    camera = cv2.VideoCapture(0)

    # region of interest (ROI) coordinates
    top, right, bottom, left = 80, 350, 225, 590

    # initialize num of frames
    num_frames = 0

    # keep looping, until interrupted
    while(True):
        # get the current frame
        (grabbed, frame) = camera.read()

        # resize the frame
        frame = imutils.resize(frame, width=700)

        # flip the frame so that it is not the mirror view
        frame = cv2.flip(frame, 1)

        # clone the frame
        clone = frame.copy()

        # get the height and width of the frame
        (height, width) = frame.shape[:2]

        # get the ROI
        roi = frame[top:bottom, right:left]

        # convert the roi to grayscale and blur it
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # to get the background, keep looking till a threshold is reached
        # so that our running average model gets calibrated
        if num_frames < 30:
            run_avg(gray, aWeight)
        else:
            # segment the hand region
            hand = segment(gray)

            # check whether hand region is segmented
            if hand is not None:
                # if yes, unpack the thresholded image and
                # segmented region
                (thresholded, segmented) = hand

                # draw the segmented region and display the frame
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                cv2.imshow("Thesholded", thresholded)

                _,contours,hierarchy= cv2.findContours(thresholded,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

                cnt = max(contours, key = lambda x: cv2.contourArea(x))
                epsilon = 0.0005*cv2.arcLength(cnt,True)
                approx= cv2.approxPolyDP(cnt,epsilon,True)
                hull = cv2.convexHull(cnt)

                areahull = cv2.contourArea(hull)
                areacnt = cv2.contourArea(cnt)
                arearatio=((areahull-areacnt)/areacnt)*100
                hull = cv2.convexHull(approx, returnPoints=False)
                defects = cv2.convexityDefects(approx, hull)

                l=0

                for i in range(defects.shape[0]):
                    s,e,f,d = defects[i,0]
                    start = tuple(approx[s][0])
                    end = tuple(approx[e][0])
                    far = tuple(approx[f][0])
                    pt= (100,180)
            
            
                    # find length of all sides of triangle
                    a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                    b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                    c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                    s = (a+b+c)/2
                    ar = math.sqrt(s*(s-a)*(s-b)*(s-c))
            
                    #distance between point and convex hull
                    d=(2*ar)/a
            
                    # apply cosine rule here
                    angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
            
        
                    # ignore angles > 90 and ignore points very close to convex hull(they generally come due to noise)
                    if angle <= 90 and d>30:
                        l += 1
                        cv2.circle(roi, far, 3, [255,0,0], -1)
            
                    #draw lines around hand
                    cv2.line(roi,start, end, [0,255,0], 2)
                l+=1
            #print corresponding gestures which are in their ranges
                #print(l)
                #print("ratio: ",arearatio)
                #print("cnt: ",areacnt)
                font = cv2.FONT_HERSHEY_SIMPLEX
                if l==1:
                    if areacnt<2000:
                        
                        cv2.putText(frame,'Put hand in the box',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                    else:
                        if arearatio<=4 and areacnt<6000:
                            cv2.putText(frame,'S',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                        elif arearatio<=4 and arearatio>0 and areacnt>=7000:
                            cv2.putText(frame,'B',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                        elif arearatio<=5 and areacnt<8500:
                            cv2.putText(frame,'E',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                        elif arearatio>11 and arearatio<16:
                            cv2.putText(frame,'A',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                        elif arearatio>=16 and arearatio<21:
                            cv2.putText(frame,'D',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA) 

                        elif arearatio>26:
                            cv2.putText(frame,'Y',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                        else:
                            cv2.putText(frame,'',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                elif l==2:
                    if arearatio<18 and areacnt<8000:
                        cv2.putText(frame,'F',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                    elif arearatio<30 and areacnt<7500:
                        cv2.putText(frame,'C',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                    else:
                        cv2.putText(frame,'V',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                elif l==3:
         
                    if arearatio<27:
                        cv2.putText(frame,'3',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                    else:
                        cv2.putText(frame,'W',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                    
                elif l==4:
                    cv2.putText(frame,'4',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            
                elif l==5:
                    cv2.putText(frame,'5',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            
                elif l==6:
                    cv2.putText(frame,'reposition',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            
                else :
                    cv2.putText(frame,'reposition',(10,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            

        # draw the segmented hand
        cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)

        # increment the number of frames
        num_frames += 1

        # display the frame with segmented hand
        cv2.imshow("Video Feed", clone)

        cv2.imshow("T",frame)

        # observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF

        # if the user pressed "q", then stop looping
        if keypress == ord("q"):
            break

        if keypress == ord("r"):
            run_avg(grey,aWeight)
            break
# free up memory
camera.release()
cv2.destroyAllWindows()