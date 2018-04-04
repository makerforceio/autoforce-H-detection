#!/usr/bin/env python

import numpy as np
import cv2 as cv

def h_detect(frame, debug=False): #Returns found_h
    mask = cv.inRange(frame, np.array([0, 0, 0]), np.array([150, 150, 150]))
    mask = cv.blur(mask, (10,10))
    mask = cv.threshold(mask, 100, 255, cv.THRESH_BINARY)[1]
    
    contours = cv.findContours(mask, 1, cv.CHAIN_APPROX_NONE)[1]
    if len(contours) > 0:
        cnt = max(contours, key=cv.contourArea)
        rect = cv.minAreaRect(cnt)
        angle = rect[2]
        if rect[1][0] > rect[1][1]:
            angle = angle + 90
        print(angle)

        rows,cols = mask.shape[:2]
        M = cv.getRotationMatrix2D((cols/2,rows/2),angle,1)
        mask = cv.warpAffine(mask,M,(cols,rows))
        
        mask = cv.convertScaleAbs(mask)
        total = np.zeros(mask.shape, np.float32)##mask.copy()
        length = 30
        for i in range(3):
            for i2 in range(length):
                M = np.float32([[1,0,0],[0,1,i*length+i2-(length * 2)/3]])
                total += cv.warpAffine(mask,M,(cols,rows))
        total = total * (255/total.max())
        total = cv.convertScaleAbs(total)
        #total = cv.blur(total, (10,10))
        total = cv.threshold(total, 200, 255, cv.THRESH_BINARY)[1]

        contours = cv.findContours(total, 1, cv.CHAIN_APPROX_NONE)[1]
        if len(contours) > 1:
            cnt, cnt2 = sorted(contours, key=cv.contourArea)[:2]
            x,y,w,h = cv.boundingRect(cnt)
            x2,y2,w2,h2 = cv.boundingRect(cnt2)
            y_min = min(y, y2)
            x_min = min(x+w, x2+w2)
            y_max = max(y+h, y2+h2)
            x_max = max(x, x2)
            print("{}:{},{}:{}".format(y_min, y_max, x_min, x_max))
            H = mask[y_min:y_max,x_min:x_max].copy()
            H = cv.convertScaleAbs(H)
            total_H = np.zeros(H.shape, np.float32)##mask.copy()
            length = 30
            rows,cols = H.shape[:2]
            for i in range(3):
                for i2 in range(length):
                    M = np.float32([[1,0,i*length+i2-(length * 2)/3],[0,1,0]])
                    total_H += cv.warpAffine(H,M,(cols,rows))
            total_H = total_H * (255/total_H.max())
            total_H = cv.convertScaleAbs(total_H)
            #total = cv.blur(total, (10,10))
            total_H = cv.threshold(total_H, 200, 255, cv.THRESH_BINARY)[1]
            contours = cv.findContours(total_H, 1, cv.CHAIN_APPROX_NONE)[1]
            if len(contours) > 0:
                cnt = max(contours, key=cv.contourArea)
                if cv.contourArea(cnt) > 10:
                    x,y,w,h = cv.boundingRect(cnt)
                    rows = float(rows)
                    cols = float(cols)
                    if (0.3 < (y + h/2)/rows < 0.7):
                        return True

    return False

if __name__ == "__main__":
    import h_detection_local
