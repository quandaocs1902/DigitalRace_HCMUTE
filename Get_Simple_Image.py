import base64
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import cv2
from PIL import Image
from flask import Flask
from io import BytesIO
from sklearn.linear_model import RANSACRegressor
import ransac as RS
#------------- Add library ------------#

#--------------------------------------#
#Global variable
MAX_SPEED = 100
MAX_ANGLE = 25
# Tốc độ thời điểm ban đầu
speed_limit = MAX_SPEED
MIN_SPEED = 10

#initialize our server
sio = socketio.Server()
#our flask (web) app
app = Flask(__name__)
#registering event handler for the server

def bird_view(img):
    # img_size = (img.shape[1], img.shape[0])
    # src = np.float32([[0, 240], [160 - 640/21, 60],
    #                  [160 + 640/21, 60], [320, 240]])

    # # cv2.imshow("img", img)
    # offset = [80, 0]
    # dst = np.float32([src[0] + offset, np.array([src[0, 0], 0]) + offset,
    #                   np.array([src[3, 0], 0]) - offset, src[3] - offset])

    # # Given src and dst points, calculate the perspective transform matrix
    # M = cv2.getPerspectiveTransform(src, dst)
    # # Warp the image using OpenCV warpPerspective()
    # birdview = cv2.warpPerspective(img, M, img_size)
    # # Return the resulting image and matrix
    # # invM = cv2.getPerspectiveTransform(dst, src)
    global birdview
    width, height = 300, 320
    pts1 = np.float32([[0, 100], [300, 100], [0, 200], [300, 200]])
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    birdview = cv2.warpPerspective(img, matrix, (height, width))
    # cv2.imshow("bird view", birdview)
    return birdview

def find_threshold(img):
    roi = img[160:190, 140:180]
    threshold = np.mean(roi)
    return threshold

# Data preprocessing
def preprocess(img):
    # Read image
    # img = cv.imread(str(image_path), cv.IMREAD_COLOR)
    birdview = bird_view(img)
    # Convert in to HSV color space
    hsv = cv2.cvtColor(birdview, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    # Blurr image
    s_blurred = cv2.medianBlur(s, 3)
    v_blurred = cv2.medianBlur(v, 3)

    # Convert to binary image
    s_threshold = find_threshold(s) + 20
    v_threshold = find_threshold(v) + 50

    _, s_thresh = cv2.threshold(s_blurred, s_threshold, 255, cv2.THRESH_BINARY_INV)
    _, v_thresh = cv2.threshold(v_blurred, v_threshold, 255, cv2.THRESH_BINARY)

    # final = cv.bitwise_and(s_thresh, v_thresh)[100:, :]
    final = cv2.bitwise_and(s_thresh, v_thresh)
    #canny = canny_edge_detect(final)
    canny = cv2.Canny(final, 150, 220)
    return canny

def downsample(ROI):
    col_index = 0
    result = np.zeros((30, 320), dtype=np.float)
    for i in range(0, ROI.shape[0], 10):
        for j in range(0, ROI.shape[1]):
            process = ROI[i: i + 10, j]
            value = np.sum(process, dtype=np.float)
            result[col_index][j] = value
        col_index += 1
    return result

# Get consecutive array
def ranges(nums):
    gaps = [[s, e]
            for s, e in zip(nums, nums[1:]) if (s + 1 < e and s + 70 < e)]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    return list(zip(edges, edges))

# Determine xLeft, xMid, xRight
def getmeanX(process, img):
    dataLeft = []
    dataRight = []
    dataMid = []
    for i in range(process.shape[0]):
        ROI = process[i, :]
        xValue = np.where(ROI > 0)
        x = ranges(list(xValue[0]))
        try:
            for j in x:
                if (np.mean(j, dtype=np.int) <= 80):
                    left = [np.mean(j, dtype=np.int), 5 + i * 10]
                    dataLeft.append(left)
                    # Tuple to draw circle
                    l = (np.mean(j, dtype = np.int), 5 + i * 10)
                    cv2.circle(birdview, l, 2, (214, 57, 17), 2)
                elif (np.mean(j, dtype=np.int) >= 160):
                    right = [np.mean(j, dtype=np.int), 5 + i * 10]
                    dataRight.append(right)
                    # Tuple to draw circle
                    r = (np.mean(j, dtype = np.int), 5 + i * 10)
                    cv2.circle(birdview, r, 2, (0, 84, 163), 2)
                else:
                    dataMid.append((np.mean(j, dtype=np.int), 5 + i * 10))
        except IndexError:
            pass
        cv2.imshow("warped", birdview)
    return [np.asarray(dataLeft, dtype=np.int), np.asarray(dataRight, dtype=np.int)]

# Using Ransac to fit lane
def ransacSklearn(data):
    data = np.sort(data, axis = 0)
    X = np.asarray(data[: , 0], dtype = np.int32)
    y = data[: , 1]

    lineX = X.reshape((len(data), 1))
    ransac = RANSACRegressor(RS.PolynomialRegression(degree = 2), residual_threshold= 2 * np.std(y), random_state=0)
    ransac.fit(lineX, y)
    lineY = ransac.predict(lineX)
    poly_Coef = np.polyfit(X, lineY, 2)
    lineY = np.polyval(poly_Coef, lineX).reshape((-1, 1))
    result = np.hstack((lineX, lineY))
    return np.int32(result)


def line_detect(img):
    process = downsample(img)
    left, right = getmeanX(process, img)
    try:
        leftCurve = ransacSklearn(left)
        cv2.polylines(img, [leftCurve], False, (255, 0, 0), 2)
    except IndexError:
        pass
    try:
        rightCurve = ransacSklearn(right)
        cv2.polylines(img, [rightCurve], False, (255, 0, 0), 2)
    except IndexError:
        pass
    return img
@sio.on('telemetry')
def telemetry(sid, data):
    if data:


        steering_angle = 0  #Góc lái hiện tại của xe
        speed_callback = 0           #Vận tốc hiện tại của xe
        image = 0           #Ảnh gốc

        steering_angle = float(data["steering_angle"])
        speed_callback = float(data["speed"])
        #Original Image
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        image = np.asarray(image)
        
        """
        - Chương trình đưa cho bạn 3 giá trị đầu vào:
            * steering_angle: góc lái hiện tại của xe
            * speed: Tốc độ hiện tại của xe
            * image: hình ảnh trả về từ xe
        
        - Bạn phải dựa vào 3 giá trị đầu vào này để tính toán và gửi lại góc lái và tốc độ xe cho phần mềm mô phỏng:
            * Lệnh điều khiển: send_control(sendBack_angle, sendBack_Speed)
            Trong đó:
                + sendBack_angle (góc điều khiển): [-25, 25]  NOTE: ( âm là góc trái, dương là góc phải)
                + sendBack_Speed (tốc độ điều khiển): [-150, 150] NOTE: (âm là lùi, dương là tiến)
        """
        sendBack_angle = 0
        sendBack_Speed = 0
        try:
            #------------------------------------------  Work space  ----------------------------------------------#
            # print(image.shape)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            cv2.imshow("image",image)
      
            cv2.waitKey(1)
            # birdview = bird_view(image)
            # cv2.waitKey(1)
            canny = preprocess(image)
            cv2.imshow("canny", canny)
            img = line_detect(canny)
            # cv2.waitKey(1)
            #------------------------------------------------------------------------------------------------------#
            # print('{} : {}'.format(sendBack_angle, sendBack_Speed))
            send_control(sendBack_angle, sendBack_Speed)
        except Exception as e:
            print(e)
    else:
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print(sid, "connect ")
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__(),
        },
        skip_sid=True)




if __name__ == '__main__':

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)
    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
    
