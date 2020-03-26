# GenData.py

import sys
import numpy as np
import cv2
import os

MIN_CONTOUR_AREA = 100
width = 20
height = 30

def main():
    img = cv2.imread("training_chars.png")

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlurred = cv2.GaussianBlur(imgGray, (5,5), 0)
    imgThresh = cv2.adaptiveThreshold(imgBlurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    _, contours, __ = cv2.findContours(imgThresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ValidChars = [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9'),
                     ord('A'), ord('B'), ord('C'), ord('D'), ord('E'), ord('F'), ord('G'), ord('H'), ord('I'), ord('J'),
                     ord('K'), ord('L'), ord('M'), ord('N'), ord('O'), ord('P'), ord('Q'), ord('R'), ord('S'), ord('T'),
                     ord('U'), ord('V'), ord('W'), ord('X'), ord('Y'), ord('Z')]

    patterns = np.empty((0, width * height))
    labels = []

    for contour in contours:
        if cv2.contourArea(contour) > MIN_CONTOUR_AREA:
            [x, y, w, h] = cv2.boundingRect(contour)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
            charArea = imgThresh[y:y+h, x:x+w]
            cv2.imshow("charArea", charArea)
            cv2.imshow("training_numbers.png", img)

            key = cv2.waitKey(0)
            if key == 27: #esc key
                sys.exit()
            elif key in ValidChars:
                labels.append(key)
                charAreaResized = cv2.resize(charArea, (width, height))
                pattern = charAreaResized.reshape((1, width*height))
                patterns = np.append(patterns, pattern, 0)

    labels = np.array(labels, np.float32)
    labels = labels.reshape((labels.size, 1))

    np.savetxt("labels.txt", labels)
    np.savetxt("patterns.txt", patterns)
    cv2.destroyAllWindows()
    return

if __name__ == "__main__":
    main()