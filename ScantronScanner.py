import cv2
import numpy as np


def preProccesing(img):
    imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGrey, (5, 5), 1)
    imgCanny = cv2.Canny(imgGrey, 50, 50)
    kern = np.ones((5, 5), np.uint8)
    imgDilate = cv2.dilate(imgCanny, kern, iterations=2)
    imgThresh = cv2.erode(imgDilate, kern, iterations=1)
    return imgThresh


def getBiggestContour( img):
    contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for cont in contours:
        p = cv2.arcLength(cont, True)
        approx = cv2.approxPolyDP(cont, 0.02 * p, True)
        if len(approx) == 4:
            return approx
    return None


def orderPts(pts):
    tr = -1;
    br = -1;
    tl = -1;
    bl = -1
    leftRightSort = pts[np.argsort(pts[:, 0]), :]
    leftPts = leftRightSort[:2]
    rightPts = leftRightSort[2:]
    tl, bl = leftPts[np.argsort(leftPts[:, 1]), :]
    tr, br = rightPts[np.argsort(rightPts[:, 1]), :]
    return np.float32([tl, tr, bl, br])


def fourPointWarp(img, pts):
    rect = orderPts(pts)
    (tl, tr, bl, br) = rect
    temp1 = np.sqrt((br[0] - bl[0]) ** 2 + (br[1] - bl[1]) ** 2)
    temp2 = np.sqrt((tr[0] - tl[0]) ** 2 + (tr[1] - tl[1]) ** 2)
    width = int(max(temp1, temp2))
    temp1 = np.sqrt((br[0] - tr[0]) ** 2 + (br[1] - tr[1]) ** 2)
    temp2 = np.sqrt((bl[0] - tl[0]) ** 2 + (bl[1] - tl[1]) ** 2)
    height = int(max(temp1, temp2))
    dest = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    m = cv2.getPerspectiveTransform(rect, dest)
    return cv2.warpPerspective(img, m, (width, height))


def scanImg(img):
    imgCont = img.copy()
    img = preProccesing(img)
    biggest = getBiggestContour(img)
    orderPts(biggest.reshape(4, 2))
    # cv2.drawContours(imgCont, biggest, -1, (255, 0, 0), 3)
    cv2.imshow("dh", imgCont)
    cv2.imshow("image", fourPointWarp(imgCont, biggest.reshape(4, 2)))


def main():
    i = cv2.imread("test3.jpg")
    scanImg(i)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
