import Functions as func
import cv2
import numpy as np
import math

img = cv2.imread('car4.png')
# hsv transform - value = gray image
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hue, saturation, value = cv2.split(hsv)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
topHat = cv2.morphologyEx(value, cv2.MORPH_TOPHAT, kernel)
blackHat = cv2.morphologyEx(value, cv2.MORPH_BLACKHAT, kernel)
add = cv2.add(value, topHat)
subtract = cv2.subtract(add, blackHat)
blur = cv2.GaussianBlur(subtract, (5, 5), 0)
thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 9)
height, width = thresh.shape

_, contours, __ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# all possible chars
possibleChars = []
for contour in contours:
    possibleChar = func.Char(contour)
    if func.checkIfChar(possibleChar) is True:
        possibleChars.append(possibleChar)

imageContours = np.zeros((height, width, 3), np.uint8)
all_contours = []
for char in possibleChars:
    all_contours.append(char.contour)
cv2.drawContours(imageContours, all_contours, -1, (255, 255, 255))
cv2.imshow('imageContours', imageContours)

# number chars in plate
listOfMatchingChars = []
for possibleC in possibleChars:
    listOfMatchingChars = func.listMatchWithChar(possibleC, possibleChars)
    # don't forget current char
    listOfMatchingChars.append(possibleC)
    # len of plate number >= 7
    if len(listOfMatchingChars) < 7:
        continue
    break

finalContours = np.zeros((height, width, 3), np.uint8)
less_contours = []
for matchingChar in listOfMatchingChars:
    less_contours.append(matchingChar.contour)
cv2.drawContours(finalContours, less_contours, -1, (255, 255, 255))
cv2.imshow("finalContours", finalContours)

# plate attribute
possiblePlate = func.PossiblePlate()
listOfMatchingChars.sort(key=lambda matchingChar: matchingChar.centerX)
plateCenterX = (listOfMatchingChars[0].centerX + listOfMatchingChars[len(listOfMatchingChars) - 1].centerX) / 2.0
plateCenterY = (listOfMatchingChars[0].centerY + listOfMatchingChars[len(listOfMatchingChars) - 1].centerY) / 2.0
plateCenter = plateCenterX, plateCenterY
plateWidth = int((listOfMatchingChars[len(listOfMatchingChars) - 1].boundingRectX + listOfMatchingChars[
    len(listOfMatchingChars) - 1].boundingRectWidth - listOfMatchingChars[0].boundingRectX) * 1.3)
totalOfCharHeights = 0
for matchingChar in listOfMatchingChars:
    totalOfCharHeights = totalOfCharHeights + matchingChar.boundingRectHeight
averageCharHeight = totalOfCharHeights / len(listOfMatchingChars)
plateHeight = int(averageCharHeight * 1.5)
opposite = listOfMatchingChars[len(listOfMatchingChars) - 1].centerY - listOfMatchingChars[0].centerY
hypotenuse = func.distanceBetweenChars(listOfMatchingChars[0], listOfMatchingChars[len(listOfMatchingChars) - 1])
correctionAngleInRad = math.asin(opposite / hypotenuse)
correctionAngleInDeg = correctionAngleInRad * (180.0 / math.pi)
possiblePlate.rrLocationOfPlateInScene = (tuple(plateCenter), (plateWidth, plateHeight), correctionAngleInDeg)
rotationMatrix = cv2.getRotationMatrix2D(tuple(plateCenter), correctionAngleInDeg, 1.0)
imgRotated = cv2.warpAffine(thresh, rotationMatrix, (width, height))
imgCropped = cv2.getRectSubPix(imgRotated, (plateWidth, plateHeight), tuple(plateCenter))
possiblePlate.Plate = imgCropped


if possiblePlate.Plate is not None:
    plate = possiblePlate
    p2fRectPoints = cv2.boxPoints(plate.rrLocationOfPlateInScene)
    rectColour = (0, 255, 0)
    cv2.line(img, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), rectColour, 2)
    cv2.line(img, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), rectColour, 2)
    cv2.line(img, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), rectColour, 2)
    cv2.line(img, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), rectColour, 2)
    cv2.imshow("plate", cv2.resize(plate.Plate, ((600, 100))))
else:
    print('No plate found')
    exit()

cv2.putText(img, func.recognizeCharsInPlate(thresh, listOfMatchingChars), (int(width/4), int(height/5)), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
cv2.imshow("final detected", img)
cv2.waitKey(0)