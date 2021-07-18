from sklearn.linear_model import RANSACRegressor
import numpy as np
import cv2 as cv
from sklearn.metrics import mean_squared_error
from numpy.lib.polynomial import polyfit
import math
def pers_transform(img):
    img_size = (img.shape[1], img.shape[0])
    src = np.float32([[0, 180], [160 - 48, 110],
                     [160 + 48, 110], [320, 180]])

    # cv2.imshow("img", img)
    offset = [75, 0]
    dst = np.float32([src[0] + offset, np.array([src[0, 0], 0]) + offset,
                      np.array([src[3, 0], 0]) - offset, src[3] - offset])

    # Given src and dst points, calculate the perspective transform matrix
    M = cv.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    birdview = cv.warpPerspective(img, M, img_size)
    return birdview
# Find average threshold of line
def find_threshold(img):
    roi = img[120:160, 140:180]
    threshold = np.mean(roi)
    return threshold

# Data preprocessing
def preprocess(img, medianKsize = 3, s_offset = 20, v_offset = 50, CannyThres1 = 100, CannyThres2 = 200):
    """
        preprocess function detect edges of the lane

        :param img: raw image got from simulation
        :param medianKsize: kernel size for Median Blur
        :param s_offset: offset to find mean threshold for s image
        :param v_offset: offset to find mean threshold for v image
        :param CannyThres1: lower threshold for Canny Edge Detection
        :param CannyThres2: higher threshold for Canny Edge Detection

        :return: noise-filtered and canny-applied ROI
    """
    # Read image
    global warped
    # img = cv.imread(str(image_path), cv.IMREAD_COLOR)
    warped = pers_transform(img)
    cv.imshow("warp", warped)
    # Convert in to HSV color space
    hsv = cv.cvtColor(warped, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)
    # Blurr image
    s_blurred = cv.medianBlur(s, medianKsize)
    v_blurred = cv.medianBlur(v, medianKsize)

    # Convert to binary image
    s_threshold = find_threshold(s) + s_offset
    v_threshold = find_threshold(v) + v_offset

    _, s_thresh = cv.threshold(s_blurred, s_threshold, 255, cv.THRESH_BINARY_INV)
    _, v_thresh = cv.threshold(v_blurred, v_threshold, 255, cv.THRESH_BINARY)
    final = cv.bitwise_and(s_thresh, v_thresh)[30:, :]
    # final = cv.bitwise_and(s_thresh, v_thresh)
    cv.imshow("s_thresh", s_thresh)
    cv.imshow("v_thresh", v_thresh)
    canny = cv.Canny(final, CannyThres1, CannyThres2)
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
            for s, e in zip(nums, nums[1:]) if (s + 1 < e and s + 30 < e)]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    return list(zip(edges, edges))

# Determine xLeft, xMid, xRight
def getmeanX(process, left_thres = 80, right_thres = 160):
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
                    cv.circle(warped, l, 2, (214, 57, 17), 2)
                elif (np.mean(j, dtype=np.int) >= right_thres):
                    right = [np.mean(j, dtype=np.int), 5 + i * 10]
                    dataRight.append(right)
                    # Tuple to draw circle
                    r = (np.mean(j, dtype = np.int), 5 + i * 10)
                    cv.circle(warped, r, 2, (0, 84, 163), 2)
                # else:
                #     dataMid.append((np.mean(j, dtype=np.int), 5 + i * 10))
        except IndexError:
            pass
        cv.imshow("warped", warped)
    return [np.asarray(dataLeft, dtype=np.int), np.asarray(dataRight, dtype=np.int)]

# Using Ransac to fit lane
def ransacSklearn(data, img):
    data = np.sort(data, axis = 0)
    X = np.asarray(data[: , 0], dtype = np.int32)
    y = data[: , 1]
    lineX = X.reshape((len(data), 1))

    ransac = RANSACRegressor()
    ransac.fit(lineX, y)
    lineY = ransac.predict(lineX)

    poly_Coef = np.polyfit(X, lineY, 1)
    lineY = np.polyval(poly_Coef, lineX).reshape((-1, 1))
    result = np.hstack((lineX, lineY))
    cv.polylines(img, [np.int32(result)], False, (255, 0, 0), 2)

    return poly_Coef

class PolynomialRegression(object):
    def __init__(self, degree = 2, coeffs = None):
        self.degree = degree
        self.coeffs = coeffs
    
    def fit(self, X, y):
        self.coeffs = np.polyfit(X.ravel(), y, self.degree)

    def get_params(self, deep = False):
        return {'coeffs': self.coeffs}
    
    def set_params(self, coeffs = None, random_state = None):
        self.coeffs = coeffs
    
    def predict(self, X):
        poly_eqn = np.poly1d(self.coeffs)
        y_hat = poly_eqn(X.ravel())
        return y_hat
    
    def score(self, X, y):
        return mean_squared_error(y, self.predict(X))

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
        rightCurve = ransacSklearn(right, img)

    except Exception:
        pass

    # try:
    #     error_angle = get_angle(leftCurve[220], rightCurve[220], 220)
    # except IndexError:
    #     error_angle = 0
    cv.imshow("line detected", img)
    return 0
