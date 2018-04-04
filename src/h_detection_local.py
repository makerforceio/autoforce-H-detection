#!/usr/bin/env python

import cv2 as cv

from h_detection import *

img = cv.imread("h3.png", cv.IMREAD_COLOR)
img = cv.resize(img, (0,0), fx=0.3, fy=0.3)

out = h_detect(img, debug=True)
cv.imshow('frame', img)
'''
for i, img in enumerate(debug):
    cv.imshow('i{}'.format(i), img)
'''
print(out)
    
while True:
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()
