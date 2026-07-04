import numpy as np
import cv2
from collections import deque

def setValues(x):
    pass

# Correct window name
cv2.namedWindow("Color detector")

# Trackbars
cv2.createTrackbar("Upper Hue","Color detector",153,180,setValues)    
cv2.createTrackbar("Upper Saturation","Color detector",255,255,setValues) 
cv2.createTrackbar("Upper Value","Color detector",255,255,setValues)

cv2.createTrackbar("Lower Hue","Color detector",64,180,setValues)
cv2.createTrackbar("Lower Saturation","Color detector",72,255,setValues)
cv2.createTrackbar("Lower Value","Color detector",49,255,setValues)

# Points storage
bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
ypoints = [deque(maxlen=1024)]

blue_index=0
red_index=0
green_index=0
yellow_index=0

kernal=np.ones((5,5),np.uint8)

colors = [(255,0,0),(0,255,0),(0,0,255),(0,255,255)]
colorIndex = 0

# Paint window
paintWindow = np.ones((471,636,3), dtype=np.uint8) * 255
cv2.rectangle(paintWindow,(40,1),(140,65),(0,0,0),2)
cv2.rectangle(paintWindow,(160,1),(255,65),colors[0],-1)
cv2.rectangle(paintWindow,(275,1),(370,65),colors[1],-1)
cv2.rectangle(paintWindow,(390,1),(485,65),colors[2],-1)
cv2.rectangle(paintWindow,(505,1),(600,65),colors[3],-1)
cv2.putText(paintWindow,"Clear",(49,33),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2,cv2.LINE_AA)

cv2.namedWindow('Paint')

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame,1)
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    # Get HSV values from trackbars
    u_hue = cv2.getTrackbarPos("Upper Hue","Color detector")
    u_sat = cv2.getTrackbarPos("Upper Saturation","Color detector")
    u_val = cv2.getTrackbarPos("Upper Value","Color detector")

    l_hue = cv2.getTrackbarPos("Lower Hue","Color detector")
    l_sat = cv2.getTrackbarPos("Lower Saturation","Color detector")
    l_val = cv2.getTrackbarPos("Lower Value","Color detector")

    upper_hsv = np.array([u_hue,u_sat,u_val])
    lower_hsv = np.array([l_hue,l_sat,l_val])

    # Draw buttons on frame
    cv2.rectangle(frame,(40,1),(140,65),(0,0,0),-1)
    cv2.rectangle(frame,(160,1),(255,65),colors[0],-1)
    cv2.rectangle(frame,(275,1),(370,65),colors[1],-1)
    cv2.rectangle(frame,(390,1),(485,65),colors[2],-1)
    cv2.rectangle(frame,(505,1),(600,65),colors[3],-1)
    cv2.putText(frame,"Clear",(49,33),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2,cv2.LINE_AA)

    mask=cv2.inRange(hsv,lower_hsv,upper_hsv)
    mask=cv2.erode(mask,kernal,iterations=1)
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernal)
    mask=cv2.dilate(mask,kernal,iterations=1)
    cv2.imshow("mask",mask)

    cnts,z=cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    center = None

    if len(cnts) > 0:
        cnt = sorted(cnts,key=cv2.contourArea,reverse=True)[0]
        ((x,y),radius) = cv2.minEnclosingCircle(cnt)
        cv2.circle(frame,(int(x),int(y)),int(radius),(0,255,255),2)
        M = cv2.moments(cnt)
        center = (int(M['m10']/M['m00']),int(M['m01']/M['m00']))

        if center[1] <= 65:
            if 40 <= center[0] <= 140:
                bpoints = [deque(maxlen=512)]
                gpoints = [deque(maxlen=512)]
                rpoints = [deque(maxlen=512)]
                ypoints = [deque(maxlen=512)]

                blue_index=0
                red_index=0
                green_index=0
                yellow_index=0

                paintWindow[67:,:,:] = 255
            elif 160<= center[0] <=255:
                colorIndex = 0
            elif 275<= center[0] <= 485:
                colorIndex = 1 
            elif 390<= center[0] <= 485:
                colorIndex = 2
            elif 505<= center[0] <= 600:
                colorIndex = 3
        else:
            if colorIndex == 0:
                bpoints[blue_index].appendleft(center)

            elif colorIndex == 1:
                gpoints[green_index].appendleft(center)
            elif colorIndex == 2:
                rpoints[red_index].appendleft(center)
            elif colorIndex == 3:
                ypoints[yellow_index].appendleft(center)

    else:
        bpoints.append(deque(maxlen=512))
        blue_index +=1
        gpoints.append(deque(maxlen=512))
        green_index +=1 
        rpoints.append(deque(maxlen=512))
        red_index +=1                                            
        ypoints.append(deque(maxlen=512))
        yellow_index +=1   

    points = [bpoints,gpoints,ypoints,rpoints]
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1,len(points[i][j])):
                if points[i][j][k-1] is None or points[i][j][k] is None:
                    continue
                cv2.line(frame,points[i][j][k-1],points[i][j][k],colors[i],2)
                cv2.line(paintWindow,points[i][j][k-1],points[i][j][k],colors[i],2)

    cv2.imshow("live drawing",frame)            
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

