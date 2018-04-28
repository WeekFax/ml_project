import imutils
from imutils import contours
import numpy as np
import cv2

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped



# define the dictionary of digit segments so we can identify
# each digit on the thermostat
DIGITS_LOOKUP = {
	(1, 1, 1, 0, 1, 1, 1): 0,
	(0, 0, 1, 0, 0, 1, 0): 1,
	(1, 0, 1, 1, 1, 1, 0): 2,
	(1, 0, 1, 1, 0, 1, 1): 3,
	(0, 1, 1, 1, 0, 1, 0): 4,
	(1, 1, 0, 1, 0, 1, 1): 5,
	(1, 1, 0, 1, 1, 1, 1): 6,
	(1, 0, 1, 0, 0, 1, 0): 7,
	(1, 1, 1, 1, 1, 1, 1): 8,
	(1, 1, 1, 1, 0, 1, 1): 9
}
cap = cv2.VideoCapture(0)
while True:

    # load the example image
    ret, image = cap.read()
    #image = cv2.imread("photo0.jpg")
    (X, Y, W, H) = (36, 26, 22, 36)

    # pre-process the image by resizing it, converting it to
    # graycale, blurring it, and computing an edge map
    image = imutils.resize(image, height=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5),1)
    edged = cv2.Canny(gray, 100, 200,5)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    displayCnt = None

    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # if the contour has four vertices, then we have found
        # the thermostat display
        if len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            if (abs((w / h) - 1.7) < 0.2)and(h > 50):
                cv2.drawContours(image, [approx], -1, (255, 0, 0), 1)
                #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)
                displayCnt = approx.reshape(4, 2)
                break

    if(displayCnt is not None):

        warped = four_point_transform(blurred, displayCnt)
        output = four_point_transform(image, displayCnt)


        warped = imutils.resize(warped, height=80)
        output = imutils.resize(output, height=80)
        thr=cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,21, 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
        thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel)


        #cv2.rectangle(output, (x, y), (x+w, y+h), (0, 0, 255), 1)

        cnts = cv2.findContours(thr.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        digitCnts = []

        #cv2.drawContours(output, cnts, -1, (255, 0, 0), 1)

        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            (x, y, w, h) = cv2.boundingRect(approx)
            if((w-W)<5 and abs(h-H)<7):
                x=(x+w)-W
                y=(y+h)-H

                digitCnts.append(approx)


        try:
            digitCnts = contours.sort_contours(digitCnts, method="left-to-right")[0]
            digits = []

            for c in digitCnts:
                # extract the digit ROI
                (x, y, w, h) = cv2.boundingRect(c)
                x = (x + w) - W
                y = (y + h) - H +2
                h = H-1
                w = W
                roi = thr[y:y + h, x:x + w]
                #cv2.rectangle(output, (x, y), (x + W, y + H), (0, 255, 0), 1)

                # compute the width and height of each of the 7 segments
                # we are going to examine
                (roiH, roiW) = roi.shape
                (dW, dH) = (int(roiW * 0.20), int(roiH * 0.15))
                dHC = int(roiH * 0.05)

                # define the set of 7 segments
                segments = [
                    ((4, 0), (w-2, dH)),  # top
                    ((2, 2), (dW + 2, h // 2)),  # top-left
                    ((w - dW-3, 2), (w-3, h // 2+2)),  # top-right
                    ((3, (h // 2) - dHC), (w-3, (h // 2) + dHC)),  # center
                    ((1, h // 2), (dW, h)),  # bottom-left
                    ((w - dW-4 , h // 2), (w-4, h)),  # bottom-right
                    ((0, h - dH), (w-4, h))  # bottom
                ]
                on = [0] * len(segments)


                for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
                    # extract the segment ROI, count the total number of
                    # thresholded pixels in the segment, and then compute
                    # the area of the segment
                    segROI = roi[yA:yB, xA:xB]
                    total = cv2.countNonZero(segROI)
                    area = (xB - xA) * (yB - yA)


                    # if the total number of non-zero pixels is greater than
                    # 50% of the area, mark the segment as "on"
                    if total / float(area) > 0.5:
                        on[i] = 1
                        cv2.rectangle(output, (xA + x, yA + y), (xB + x, yB + y), (255, 0, 0), 1)
                    else:
                        cv2.rectangle(output, (xA + x, yA + y), (xB + x, yB + y), (255, 0, 255), 1)
                        pass

                    # lookup the digit and draw it on the image

                digit = DIGITS_LOOKUP[tuple(on)]
                digits.append(digit)
                #cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)
                #cv2.putText(output, str(digit), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

            print(digits)

        except:
            print("err")


        cv2.imshow('out', output)

    cv2.imshow('result', image)

    #while True:
        #cv2.imshow('result', thr)
    ch = cv2.waitKey(5)
    if ch == 27:
        break
cv2.destroyAllWindows()
