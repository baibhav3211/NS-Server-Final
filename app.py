
# from PIL import Image
from flask import Flask, request
import cv2
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
import time
# module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"

# detector = hub.load(module_handle).signatures['default']

app = Flask(__name__)
# module_handle = r"C:\\Users\ACER\Downloads\\openimages_v4_ssd_mobilenet_v2_1"
module_handle = "./openimages_v4_ssd_mobilenet_v2_1"
detector = hub.load(module_handle)
# Define the endpoint for receiving image data
@app.route('/image', methods=['POST'])
def upload_image():
    # Decode the image data from the request body
    image_bytes = request.get_data()
    image_np = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    # print(image)
    x=detect(image)
    print(x[1])
    return x[0]

    
    # Save the image to a file
    # file_name = 'image.jpg'
    # cv2.imwrite(file_name, image)
# x
    # return 'Image saved successfully.'

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')


# print("first")

def detect(image):
 with tf.device('/GPU:0'):
    # print("hi")
    # module_handle = r"C:\\Users\ACER\Downloads\\openimages_v4_ssd_mobilenet_v2_1"

    # def run_detector(image):
    # detector = hub.load(module_handle)
    # print(image)

    converted_img  = tf.image.convert_image_dtype(image, tf.float32)[tf.newaxis, ...]
    start_time = time.time()
    result = detector.signatures['default'](converted_img)
    end_time = time.time()
    result = {key:value.numpy() for key,value in result.items()}
    return [result['detection_class_entities'][0], end_time-start_time, True]

    # print("Found %d objects." % len(result["detection_scores"]))
    # print("Inference time: ", end_time-start_time)

    # print(result['detection_class_entities'][0])


    # video_capture = cv2.VideoCapture('temp2.mp4')


    # while True:
    #     # Read the next frame
    #     ret, frame = video_capture.read()
    #     # print(ret, frame)
    #     # Check if the frame was successfully read
    #     if not ret:
    #         break
    #     # print('hello')
    #     run_detector(frame)
    #     # print('hi')
        
    #     # Wait for the user to press 'q' to exit
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    # # Release the camera and close all windows
    # video_capture.release()
    # cv2.destroyAllWindows()

