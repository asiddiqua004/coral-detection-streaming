import argparse
import collections
import common
import cv2
import numpy as np
import os
from PIL import Image
from time import sleep
import re
import tflite_runtime.interpreter as tflite
from flask import Flask, render_template, Response

Object = collections.namedtuple('Object', ['id', 'score', 'bbox'])
default_model_dir = './all_models'
default_model = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
model = os.path.join(default_model_dir,default_model)
labels = os.path.join(default_model_dir,'coco_labels.txt')
camera_idx = 0
top_k = 1
threshold = 0.1 

def load_labels(path):
    p = re.compile(r'\s*(\d+)(.+)')
    with open(path, 'r', encoding='utf-8') as f:
       lines = (p.match(line).groups() for line in f.readlines())
       return {int(num): text.strip() for num, text in lines}

class BBox(collections.namedtuple('BBox', ['xmin', 'ymin', 'xmax', 'ymax'])):
    """Bounding box.
    Represents a rectangle which sides are either vertical or horizontal, parallel
    to the x or y axis.
    """
    __slots__ = ()

def get_output(interpreter, score_threshold, top_k, image_scale=1.0):
    """Returns list of detected objects."""
    boxes = common.output_tensor(interpreter, 0)
    class_ids = common.output_tensor(interpreter, 1)
    scores = common.output_tensor(interpreter, 2)
    count = int(common.output_tensor(interpreter, 3))

    def make(i):
        ymin, xmin, ymax, xmax = boxes[i]
        return Object(
            id=int(class_ids[i]),
            score=scores[i],
            bbox=BBox(xmin=np.maximum(0.0, xmin),
                      ymin=np.maximum(0.0, ymin),
                      xmax=np.minimum(1.0, xmax),
                      ymax=np.minimum(1.0, ymax)))

    return [make(i) for i in range(top_k) if scores[i] >= score_threshold]

class Camera(object):
    def __init__(self):
        print('Loading {} with {} labels.'.format(model, labels))
        self.interpreter = common.make_interpreter(model)
        self.interpreter.allocate_tensors()
        self.labels = load_labels(labels)
        self.video = cv2.VideoCapture(0)
        self.file = open("/home/mendel/person_detected.txt","w")
    
    def __del__(self):
        self.video.release()

    def get_frame(self):
        if self.video.isOpened() :
            ret, frame = self.video.read()
            cv2_im = frame
            cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
            pil_im = Image.fromarray(cv2_im_rgb)
            common.set_input(self.interpreter, pil_im)
            self.interpreter.invoke()
            objs = get_output(self.interpreter, score_threshold=threshold, top_k=top_k)
            cv2_im = append_objs_to_img(cv2_im, objs, self.labels)

            # cv2.imshow('frame', cv2_im)
            for result in objs:
                label = '{:.0f}% {}'.format(100*result.score, self.labels.get(result.id, result.id))
                if self.labels.get(result.id) == "person" and result.score > 0.6:
                    self.file.write("1")
                    self.file.seek(0)
                else:
                    self.file.write("0")
                    self.file.seek(0) 
                print(label)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return
            sleep(0.2)
            return frame

    def get_jpeg_frame(self):
        frame = self.get_frame()
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

def main():
    camera = Camera()
    count = 0
    while count < 30 :
        camera.get_frame()
        count = count + 1

def append_objs_to_img(cv2_im, objs, labels):
    height, width, channels = cv2_im.shape
    for obj in objs:
        x0, y0, x1, y1 = list(obj.bbox)
        x0, y0, x1, y1 = int(x0*width), int(y0*height), int(x1*width), int(y1*height)
        percent = int(100 * obj.score)
        label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))

        cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2_im = cv2.putText(cv2_im, label, (x0, y0+30),
                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    return cv2_im

app = Flask(__name__)

@app.route('/')
def index():
    # rendering webpage
    return render_template('index.html')

def gen(camera):
    while True:
        #get camera frame
        frame = camera.get_jpeg_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    context = ('server.crt', 'server.key')  # certificate and key files    
    #app.run(host='0.0.0.0',debug=True, ssl_context=context)
    app.run(host='192.168.50.20',port='5000', debug=True, ssl_context=context)

