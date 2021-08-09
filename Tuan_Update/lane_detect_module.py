from sklearn.linear_model import RANSACRegressor
import numpy as np
import cv2 as cv
from sklearn.metrics import mean_squared_error
from numpy.lib.polynomial import polyfit
import math


def pers_transform(img):
    img_size = (img.shape[1], img.shape[0])
    src = np.float32([[0, 90], [160 - 352/7, 30],
                     [160 + 352/7, 30], [320, 90]])

    # cv2.imshow("img", img)
    offset = [85, 0]
    dst = np.float32([src[0] + offset, np.array([src[0, 0], 0]) + offset,
                      np.array([src[3, 0], 0]) - offset, src[3] - offset])

    # Given src and dst points, calculate the perspective transform matrix
    M = cv.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    birdview = cv.warpPerspective(img, M, img_size)
    return birdview
# Find average threshold of line


def find_threshold(img):
    roi = img[40:70, 145:175]
    threshold = np.mean(roi)
    return threshold

# Data preprocessing


def preprocess(img, alpha=1, beta=0, medianKsize=3, s_offset=20, v_offset=50, CannyThres1=100, CannyThres2=200):
    """
        preprocess function detect edges of the lane

        :param img: raw image got from simulation
        :param alpha: contrast control
        :param beta: brightness control
        :param medianKsize: kernel size for Median Blur
        :param s_offset: offset to find mean threshold for s image
        :param v_offset: offset to find mean threshold for v image
        :param CannyThres1: lower threshold for Canny Edge Detection
        :param CannyThres2: higher threshold for Canny Edge Detection

        :return: noise-filtered and canny-applied ROI
    """
    # Read image
    global warped
    
    # Convert in to HSV color space
    roi = img[90:, :]
    # cv.circle(roi, (145, 40), 2, (255, 0, 0), 2)
    # cv.circle(roi, (175, 40), 2, (255, 0, 0), 2)
    # cv.circle(roi, (145, 70), 2, (255, 0, 0), 2)
    # cv.circle(roi, (175, 70), 2, (255, 0, 0), 2)
    
    # cv.imwrite("image.jpg", roi)
    cv.imshow("goc", roi)

    # roi = cv.convertScaleAbs(roi, alpha = alpha, beta = beta)

    hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)

    # v = cv.convertScaleAbs(v, alpha = alpha, beta = beta)
    
    v = cv.equalizeHist(v)
    # Blurr image

    # s_blurred = cv.medianBlur(s, medianKsize)
    # v_blurred = cv.medianBlur(v, medianKsize)
    # Convert to binary image
    s_threshold = find_threshold(s) + s_offset
    v_threshold = find_threshold(v) + v_offset
    # print(v_threshold)
    _, s_thresh = cv.threshold(s, s_threshold, 255, cv.THRESH_BINARY_INV)
    _, v_thresh = cv.threshold(v, 230, 255, cv.THRESH_BINARY)
    # final = cv.bitwise_and(s_thresh, v_thresh)[70:, :]
    final = cv.bitwise_and(s_thresh, v_thresh)
    # kernel = np.ones((3,3),np.uint8)
    # final = cv.dilate(final, kernel, iterations = 1)
    
    cv.imshow("v", v)
    # cv.imshow("s", s)
    # cv.imshow("s_thresh", s_thresh)
    # cv.imshow("v_thresh", v_thresh)
    cv.imshow("final.", final)
    warped = pers_transform(final)
    kernel = np.ones((3,3), np.uint8)
    warped = cv.morphologyEx(warped, cv.MORPH_OPEN, kernel)
    canny = cv.Canny(warped, CannyThres1, CannyThres2)
    # cv.imshow("warped", warped)
    return canny


def downsample(ROI):
    col_index = 0
    m = ROI.shape[0]
    n = ROI.shape[1]
    result = np.zeros((m // 10, n), dtype=np.float)
    for i in range(0, m, 10):
        for j in range(0, n):
            process = ROI[i: i + 10, j]
            value = np.sum(process, dtype=np.float)
            result[col_index][j] = value
        col_index += 1
    return result

# Get consecutive array


def ranges(nums):
    gaps = [[s, e]
            for s, e in zip(nums, nums[1:]) if (s + 15 < e and s + 30 < e)]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    return list(zip(edges, edges))

# Determine xLeft, xMid, xRight


def getmeanX(process, left_thres=80, right_thres=220):
    dataLeft = []
    dataRight = []
    # dataMid = []
    for i in range(process.shape[0]):
        ROI = process[i, :]
        xValue = np.where(ROI > 0)
        x = ranges(list(xValue[0]))
        try:
            for j in x:
                if (np.mean(j, dtype=np.int) <= left_thres):
                    left = [np.mean(j, dtype=np.int), 5 + i * 10]
                    dataLeft.append(left)
                    # Tuple to draw circle
                    l = (np.mean(j, dtype = np.int), 5 + i * 10)
                    cv.circle(warped, l, 2, (255, 0, 0), 2)
                elif (np.mean(j, dtype=np.int) >= right_thres):
                    right = [np.mean(j, dtype=np.int), 5 + i * 10]
                    dataRight.append(right)
                    # Tuple to draw circle
                    r = (np.mean(j, dtype = np.int), 5 + i * 10)
                    cv.circle(warped, r, 2, (255, 0, 0), 2)
                # else:
                #     dataMid.append((np.mean(j, dtype=np.int), 5 + i * 10))
        except IndexError:
            pass
        cv.imshow("warped", warped)
    return [np.asarray(dataLeft, dtype=np.int), np.asarray(dataRight, dtype=np.int)]


class PolynomialRegression(object):
    def __init__(self, degree=2, coeffs=None):
        self.degree = degree
        self.coeffs = coeffs

    def fit(self, X, y):
        self.coeffs = np.polyfit(X.ravel(), y, self.degree)

    def get_params(self, deep=False):
        return {'coeffs': self.coeffs}

    def set_params(self, coeffs=None, random_state=None):
        self.coeffs = coeffs

    def predict(self, X):
        poly_eqn = np.poly1d(self.coeffs)
        y_hat = poly_eqn(X.ravel())
        return y_hat

    def score(self, X, y):
        return mean_squared_error(y, self.predict(X))

# Using Ransac to fit lane


def ransacSklearn(data, img, order=0):
    data = np.sort(data, axis=0)
    X = np.asarray(data[:, 0], dtype=np.int32)
    y = data[:, 1]
    if(order == 1):
        pass
    else:
        X = np.flipud(X)
        # y = np.flipud(y)

    lineX = X.reshape((len(data), 1))
    ransac = RANSACRegressor(PolynomialRegression(degree=3), residual_threshold=2 * np.std(y), random_state=0)
    ransac.fit(lineX, y)
    lineY = ransac.predict(lineX)
    poly_Coef = np.polyfit(X, lineY, 2)
    lineY = np.polyval(poly_Coef, lineX).reshape((-1, 1))
    result = np.hstack((lineX, lineY))
    cv.polylines(img, [np.int32(result)], False, (255, 0, 0), 2)

    return poly_Coef


def get_angle(xLeft, xRight, y):
    xMid = (xLeft + xRight) // 2
    error_angle = math.atan((xMid - 160) // 2)
    return error_angle


def line_detect(img):
    process = downsample(img)
    left, right = getmeanX(process)
    try:
        leftCurve = ransacSklearn(left, img)
    except Exception:
        pass
    try:
        rightCurve = ransacSklearn(right, img, order = 1)

    except Exception:
        pass

    # try:
    #     error_angle = get_angle(leftCurve[220], rightCurve[220], 220)
    # except IndexError:
    #     error_angle = 0
    cv.imshow("line detected", img)
    return 0
