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
import math
import lane_detect_module
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
        sendBack_Speed = 10
        try:
            #------------------------------------------  Work space  ----------------------------------------------#
            # print(image.shape)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow("image", image)
            cv2.imwrite("image.jpg", image)
            canny = lane_detect_module.preprocess(image, medianKsize = 3, s_offset = 20, 
                                                  v_offset = 60, CannyThres1 = 100, CannyThres2 = 200)
            cv2.imshow("canny", canny)
            a = lane_detect_module.line_detect(canny)
            cv2.waitKey(1)
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
    