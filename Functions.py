import math
import cv2
import numpy as np

class Char:
    def __init__(self, contour):
        self.contour = contour
        self.boundingRect = cv2.boundingRect(self.contour)
        [x, y, w, h] = self.boundingRect
        self.boundingRectX = x
        self.boundingRectY = y
        self.boundingRectWidth = w
        self.boundingRectHeight = h
        self.boundingRectArea = self.boundingRectWidth * self.boundingRectHeight
        self.centerX = (self.boundingRectX + self.boundingRectX + self.boundingRectWidth) / 2
        self.centerY = (self.boundingRectY + self.boundingRectY + self.boundingRectHeight) / 2
        self.diagonalSize = math.sqrt((self.boundingRectWidth ** 2) + (self.boundingRectHeight ** 2))
        self.aspectRatio = float(self.boundingRectWidth) / float(self.boundingRectHeight)

class PossiblePlate:
    def __init__(self):
        self.Plate = None
        self.Grayscale = None
        self.Thresh = None
        self.rrLocationOfPlateInScene = None
        self.strChars = ""

def checkIfChar(possibleChar):
    if (possibleChar.boundingRectArea > 80 and possibleChar.boundingRectWidth > 2
            and possibleChar.boundingRectHeight > 8 and 0.25 < possibleChar.aspectRatio < 1.0):
        return True
    else:
        return False


def distanceBetweenChars(firstChar, secondChar):
    x = abs(firstChar.centerX - secondChar.centerX)
    y = abs(firstChar.centerY - secondChar.centerY)
    return math.sqrt((x ** 2) + (y ** 2))

def angleBetweenChars(firstChar, secondChar):
    adjacent = float(abs(firstChar.centerX - secondChar.centerX))
    opposite = float(abs(firstChar.centerY - secondChar.centerY))
    if adjacent != 0.0:
        angleInRad = math.atan(opposite / adjacent)
    else:
        angleInRad = 1.5708
    angleInDeg = angleInRad * (180.0 / math.pi)
    return angleInDeg

def listMatchWithChar(possibleC, possibleChars):
    listOfMatchingChars = []
    for possibleMatchingChar in possibleChars:
        if possibleMatchingChar == possibleC:
            continue
        distance2Chars = distanceBetweenChars(possibleC, possibleMatchingChar)
        angle2Chars = angleBetweenChars(possibleC, possibleMatchingChar)
        changeInArea = float(abs(possibleMatchingChar.boundingRectArea - possibleC.boundingRectArea)) / float(possibleC.boundingRectArea)
        changeInWidth = float(abs(possibleMatchingChar.boundingRectWidth - possibleC.boundingRectWidth)) / float(possibleC.boundingRectWidth)
        changeInHeight = float(abs(possibleMatchingChar.boundingRectHeight - possibleC.boundingRectHeight)) / float(possibleC.boundingRectHeight)
        if distance2Chars < (possibleC.diagonalSize * 5) and \
                angle2Chars < 12.0 and \
                changeInArea < 0.5 and \
                changeInWidth < 0.8 and \
                changeInHeight < 0.2:
            listOfMatchingChars.append(possibleMatchingChar)
    return listOfMatchingChars

def recognizeCharsInPlate(imgThresh, listOfMatchingChars):
    knn = cv2.ml.KNearest_create()
    labels = np.loadtxt("labels.txt", np.float32)
    patterns = np.loadtxt("patterns.txt", np.float32)
    labels = labels.reshape((1, -1))
    knn.train(patterns, cv2.ml.ROW_SAMPLE, labels)
    result = ""
    listOfMatchingChars.sort(key = lambda matchingChar: matchingChar.centerX)
    for cChar in listOfMatchingChars:
        charArea = imgThresh[cChar.boundingRectY: cChar.boundingRectY + cChar.boundingRectHeight,
                           cChar.boundingRectX: cChar.boundingRectX + cChar.boundingRectWidth]
        pattern = np.float32(cv2.resize(charArea, (20, 30)).reshape(1, 20*30))
        _, res, __, ___ = knn.findNearest(pattern, k = 1)
        char = str(chr(int(res[0][0])))
        result += char
    return result