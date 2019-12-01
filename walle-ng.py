#!/usr/bin/env python
'''
 Copyright (c) 2018 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
'''

import os
import logging as log
import os.path as osp
import sys
import time
import datetime
import configparser
import cv2
import numpy as np

from openvino.inference_engine import IENetwork
from modules.ie_module import InferenceContext
from modules.landmarks_detector import LandmarksDetector
from modules.face_detector import FaceDetector
from modules.faces_database import FacesDatabase
from modules.face_identifier import FaceIdentifier
from modules.face_notification import FaceNotification

import busio
from board import SCL, SDA
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo

class FrameProcessor:
    QUEUE_SIZE = 16

    def __init__(self, config):
                
        # Get variables from configuration 
        self.device_fd = config.get('Inference', 'device_fd')
        self.device_lm = config.get('Inference', 'device_lm')
        self.device_rd = config.get('Inference', 'device_rd')
        self.thresh_fd = config.getfloat('Inference', 'thresh_fd')
        self.thresh_rd = config.getfloat('Inference', 'thresh_rd')
        self.scale_ratio = config.getfloat('Inference', 'scale_ratio')
        self.do_stats = config.getboolean('Inference', 'do_stats')
        self.do_grow = config.getboolean('Inference', 'do_grow')
        
        self.model_fd = config.get('Model', 'model_fd')
        self.model_ld = config.get('Model', 'model_ld')
        self.model_rd = config.get('Model', 'model_rd')
        
        self.db_path = config.get('Faces', 'db_path')
        self.do_detector = config.getboolean('Faces', 'do_detector')
        self.display = config.getboolean('General', 'do_output')

        # API Gateway endpoint for notifications
        self.url = config.get('AWS', 'notify_url')
        self.notify = config.getboolean('AWS', 'do_notify')
        self.api_key = config.get('AWS', 'api_key')

        used_devices = set([self.device_fd, self.device_lm, self.device_rd])
        self.context = InferenceContext()
        context = self.context
        context.load_plugins(used_devices)
        for d in used_devices:
            context.get_plugin(d).set_config({
                'PERF_COUNT': 'YES' if self.do_stats else 'NO'})

        log.info("Loading models")
        face_detector_net = self.load_model(self.model_fd)
        landmarks_net = self.load_model(self.model_ld)
        face_reid_net = self.load_model(self.model_rd)

        self.face_detector = FaceDetector(face_detector_net,
                                          confidence_threshold=self.thresh_fd,
                                          roi_scale_factor=self.scale_ratio)
        self.landmarks_detector = LandmarksDetector(landmarks_net)
        self.face_identifier = FaceIdentifier(face_reid_net,
                                              match_threshold=self.thresh_rd)

        self.face_detector.deploy(self.device_fd, context)
        self.landmarks_detector.deploy(self.device_lm, context,
                                       queue_size=self.QUEUE_SIZE)
        self.face_identifier.deploy(self.device_rd, context,
                                    queue_size=self.QUEUE_SIZE)
        log.info('Loaded models')

        log.info('Building faces database using images from {}'.format(self.db_path))
        self.faces_database = FacesDatabase(self.db_path, self.face_identifier,
                                            self.landmarks_detector,
                                            self.face_detector if self.do_detector else None, self.display)
        self.face_identifier.set_faces_database(self.faces_database)
        log.info('Database was built, registered {} identities'.format(len(self.faces_database)))

        if self.notify:
            self.face_notifier = FaceNotification(self.url)
            self.face_notifier.set_faces_database(self.faces_database)
            self.face_notifier.set_api_key(self.api_key)
            log.info('Loaded Notifier')

        self.allow_grow = self.do_grow and not self.display

    def load_model(self, model_path):
        model_path = osp.abspath(model_path)
        model_description_path = model_path
        model_weights_path = osp.splitext(model_path)[0] + ".bin"
        log.info("Loading the model from '%s'" % (model_description_path))
        assert osp.isfile(model_description_path), \
            "Model description is not found at '%s'" % (model_description_path)
        assert osp.isfile(model_weights_path), \
            "Model weights are not found at '%s'" % (model_weights_path)
        model = IENetwork(model_description_path, model_weights_path)
        log.info("Model is loaded")
        return model

    def process(self, frame):
        assert len(frame.shape) == 3, 'Expected input frame in (H, W, C) format'
        assert frame.shape[2] in [3, 4], 'Expected BGR or BGRA input'

        orig_image = frame.copy()
        frame = frame.transpose((2, 0, 1))  # HWC to CHW
        frame = np.expand_dims(frame, axis=0)

        self.face_detector.clear()
        self.landmarks_detector.clear()
        self.face_identifier.clear()
        self.face_notifier.clear()

        self.face_detector.start_async(frame)
        rois = self.face_detector.get_roi_proposals(frame)
        if self.QUEUE_SIZE < len(rois):
            log.warning('Too many faces for processing. Processing {} of {}'.format(self.QUEUE_SIZE, len(rois))) 
            rois = rois[:self.QUEUE_SIZE]
        self.landmarks_detector.start_async(frame, rois)
        landmarks = self.landmarks_detector.get_landmarks()

        self.face_identifier.start_async(frame, rois, landmarks)
        face_identities, unknowns = self.face_identifier.get_matches()
        
        if self.notify and (len(face_identities) > 0 or len(unknowns) > 0):
            self.face_notifier.start_async(face_identities, unknowns)

        if self.allow_grow and len(unknowns) > 0:
            for i in unknowns:
                # This check is preventing asking to save half-images in the boundary of images
                if rois[i].position[0] == 0.0 or rois[i].position[1] == 0.0 or \
                    (rois[i].position[0] + rois[i].size[0] > orig_image.shape[1]) or \
                        (rois[i].position[1] + rois[i].size[1] > orig_image.shape[0]):
                    continue
                crop = orig_image[int(rois[i].position[1]):int(
                    rois[i].position[1]+rois[i].size[1]), int(rois[i].position[0]):int(rois[i].position[0]+rois[i].size[0])]
                name = self.faces_database.ask_to_save(crop)
                if name:
                    id = self.faces_database.dump_faces(
                        crop, face_identities[i].descriptor, name)
                    face_identities[i].id = id

        outputs = [rois, landmarks, face_identities]

        return outputs

    def get_performance_stats(self):
        stats = {
            'face_detector': self.face_detector.get_performance_stats(),
            'landmarks': self.landmarks_detector.get_performance_stats(),
            'face_identifier': self.face_identifier.get_performance_stats(),
        }
        return stats


class Visualizer:
    BREAK_KEY_LABELS = 'q(Q) or Escape'
    BREAK_KEYS = {ord('q'), ord('Q'), 27}

    def __init__(self, config):
        
        # General Variables from config file
        self.input_video = config.get('General', 'input_video')
        self.output_video = config.get('General', 'output_video')
        self.crop_width = config.getint('General', 'crop_width')
        self.crop_height = config.getint('General', 'crop_height')    
        self.do_timelapse = config.getboolean('General', 'do_timelapse')
  
        self.frame_processor = FrameProcessor(config)
        self.display = config.getboolean('General', 'do_output')
        self.print_perf_stats = config.getboolean('Inference', 'do_stats')
        
        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0
        self.frame_num = 0
        self.frame_count = -1

        # Set up servos
        self.pca_frequency = 50
        self.start_x = 90
        self.start_y = 110
        self.x_position = self.start_x
        self.y_position = self.start_y

        # Set frame buffer device
        os.putenv('SDL_FBDEV', '/dev/fb0')

        # Set up i2c bus
        self.i2c = busio.I2C(SCL, SDA)

        # Create a simple PCA9685 class instance.
        self.pca = PCA9685(self.i2c)

        # Set freq -> use calibration.py -> clock_speed dependent
        # pca = PCA9685(i2c, reference_clock_speed=25630710)
        self.pca.frequency = self.pca_frequency

        # Specify pulse range
        # servo1 = servo.Servo(pca.channels[7], actuation_range=135)
        self.servo1 = servo.Servo(
            self.pca.channels[1], min_pulse=575, max_pulse=2325)
        self.servo0 = servo.Servo(
            self.pca.channels[0], min_pulse=575, max_pulse=2325)

        # Set the servo to (start_x,start_y)
        print('[INFO] Centering Servos')
        self.servo1.angle = self.start_x
        self.servo0.angle = self.start_y

        self.input_crop = None
        if self.crop_width and self.crop_height:
            self.input_crop = np.array((self.crop_width, self.crop_height))

        self.frame_timeout = 0 if self.do_timelapse else 1

    def update_fps(self):
        now = time.time()
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now

    def draw_text_with_background(self, frame, text, origin,
                                  font=cv2.FONT_HERSHEY_SIMPLEX, scale=1.0,
                                  color=(0, 0, 0), thickness=1, bgcolor=(255, 255, 255)):
        
        text_size, baseline = cv2.getTextSize(text, font, scale, thickness)
        cv2.rectangle(frame,
                      tuple((origin + (0, baseline)).astype(int)),
                      tuple(
                          (origin + (text_size[0], -text_size[1])).astype(int)),
                      bgcolor, cv2.FILLED)
        cv2.putText(frame, text,
                    tuple(origin.astype(int)),
                    font, scale, color, thickness)
        return text_size, baseline

    def draw_detection_roi(self, frame, roi, identity):
        label = self.frame_processor \
            .face_identifier.get_identity_label(identity.id)

        # Draw face ROI border
        cv2.rectangle(frame,
                      tuple(roi.position), tuple(roi.position + roi.size),
                      (0, 220, 0), 2)

        # Draw identity label
        text_scale = 0.5
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize('H1', font, text_scale, 1)
        line_height = np.array([0, text_size[0][1]])
        text = label
        if identity.id != FaceIdentifier.UNKNOWN_ID:
            text += ' %.2f%%' % (100.0 * (1 - identity.distance))
            self.draw_text_with_background(
                frame, text, roi.position - line_height * 0.5, font, scale=text_scale)

    def draw_detection_keypoints(self, frame, roi, landmarks):
        keypoints = [landmarks.left_eye,
                     landmarks.right_eye,
                     landmarks.nose_tip,
                     landmarks.left_lip_corner,
                     landmarks.right_lip_corner]

        for point in keypoints:
            center = roi.position + roi.size * point
            cv2.circle(frame, tuple(center.astype(int)), 2, (0, 255, 255), 2)

        # Get frame center
        rows, cols, _ = frame.shape
        center_y = int(cols / 2)
        center_x = int(rows / 2)
  
        # Calculate center     
        x_center = int((tuple(roi.position)[0] + tuple(roi.position+roi.size)[0]) / 2)
        y_center = int((tuple(roi.position)[1] + tuple(roi.position+roi.size)[1]) / 2)

        # Adjust the servos to keep the face center
        self.servo1.angle = self.get_x_position(x_center, center_x)
        self.servo0.angle = self.get_y_position(y_center, center_y)   

    def draw_detections(self, frame, detections):
        for roi, landmarks, identity in zip(*detections):
            self.draw_detection_roi(frame, roi, identity)
            self.draw_detection_keypoints(frame, roi, landmarks)

    def draw_status(self, frame, detections):
        origin = np.array([10, 10])
        color = (10, 160, 10)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_scale = 0.5
        text_size, _ = self.draw_text_with_background(
                frame, 
                'Frame time: %.3fs'.format(
                        self.frame_time),
                        origin, font, text_scale, color)
                
        self.draw_text_with_background(frame, 'FPS: {0:20.1f}'.format(self.fps),
                                       (origin + (0, text_size[1] * 1.5)), 
                                       font, text_scale, color)

        log.debug('Frame: %s/%s, detections: {} frame time: %.3fs, fps: %.1f'.format
                (self.frame_num, self.frame_count, len(detections[-1]), 
                    self.frame_time, self.fps))

        if self.print_perf_stats:
            log.info('Performance stats:')
            log.info(self.frame_processor.get_performance_stats())

    def display_interactive_window(self, frame):
        color = (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_scale = 0.5
        text = 'Press [{}] key to exit'.format(self.BREAK_KEY_LABELS)
        thickness = 2
        text_size = cv2.getTextSize(text, font, text_scale, thickness)
        origin = np.array([frame.shape[-2] - text_size[0][0] - 10, 10])
        line_height = np.array([0, text_size[0][1]]) * 1.5
        cv2.putText(frame, text,
                    tuple(origin.astype(int)), font, text_scale, color, thickness)

        cv2.imshow('Face tracking and recognition', frame)

    def should_stop_display(self):
        key = cv2.waitKey(self.frame_timeout) & 0xFF
        return key in self.BREAK_KEYS

    def process(self, input_stream, output_stream):
        self.input_stream = input_stream
        self.output_stream = output_stream

        while input_stream.isOpened():
            has_frame, frame = input_stream.read()
            if not has_frame:
                break

            if self.input_crop is not None:
                frame = Visualizer.center_crop(frame, self.input_crop)
            detections = self.frame_processor.process(frame)

            self.draw_detections(frame, detections)
            self.draw_status(frame, detections)

            if output_stream:
                output_stream.write(frame)
            if self.display:
                self.display_interactive_window(frame)
                if self.should_stop_display():
                    break

            self.update_fps()
            self.frame_num += 1

    @staticmethod
    def center_crop(frame, crop_size):
        fh, fw, fc = frame.shape
        crop_size[0] = min(fw, crop_size[0])
        crop_size[1] = min(fh, crop_size[1])
        return frame[(fh - crop_size[1]) // 2: (fh + crop_size[1]) // 2,
                     (fw - crop_size[0]) // 2: (fw + crop_size[0]) // 2,
                     :]

    def run(self):
        input_stream = Visualizer.open_input_stream(self.input_video)
        if input_stream is None or not input_stream.isOpened():
            log.error("Cannot open input stream: %s" % self.input_video)
        fps = input_stream.get(cv2.CAP_PROP_FPS)
        frame_size = (int(input_stream.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      int(input_stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.frame_count = int(input_stream.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.crop_width and self.crop_height:
            crop_size = (self.crop_width, self.crop_height)
            frame_size = tuple(np.minimum(frame_size, crop_size))
        log.info("Input stream info: %d x %d @ %.2f FPS" %
                 (frame_size[0], frame_size[1], fps))
        output_stream = Visualizer.open_output_stream(
            self.output_video, fps, frame_size)

        self.process(input_stream, output_stream)

        # Release resources
        if output_stream:
            output_stream.release()
        if input_stream:
            input_stream.release()

        cv2.destroyAllWindows()
        self.pca.deinit()

    '''Determine and adjust the x-axis of the camera - servo1'''
    def get_x_position(self, x_center, center):

        if self.x_position < 180 and self.x_position > 0:
            if x_center < center - 70:
                self.x_position += 2
            elif x_center > center + 70:
                self.x_position -= 2
        elif self.x_position >= 180:
            if x_center > center + 70:
                self.x_position -= 2
        elif self.x_position <= 0:
            if x_center < center - 70:
                self.x_position += 2

        return self.x_position


    '''Determine and adjust the y-axis of the camera - servo0'''
    def get_y_position(self, y_center, center):

        if self.y_position < 180 and self.y_position > 60:
            if y_center < center - 70:
                self.y_position -= 2
            elif y_center > center + 70:
                self.y_position += 2
        elif self.y_position >= 180:
            if y_center < center - 70:
                self.y_position -= 2
        elif self.y_position <= 60:
            if y_center > center + 70:
                self.y_position += 2

        return self.y_position

    @staticmethod
    def open_input_stream(path):
        log.info("Reading input data from '%s'" % (path))
        stream = path
        try:
            stream = int(path)
        except ValueError:
            pass
        return cv2.VideoCapture(stream)

    @staticmethod
    def open_output_stream(path, fps, frame_size):
        output_stream = None
        if path != "":
            if not path.endswith('.avi'):
                log.warning("Output file extension is not 'avi'. "
                            "Some issues with output can occur, check logs.")
            log.info("Writing output to '%s'" % (path))
            output_stream = cv2.VideoWriter(path,
                                            cv2.VideoWriter.fourcc(*'MJPG'), fps, frame_size)
        return output_stream


def main():
    config = configparser.RawConfigParser()
    config.read('config/config.cfg')
    log_file = config.get('General', 'log_file')
    
    log.basicConfig(filename=log_file, filemode='w', format="[ %(levelname)s ] %(asctime)-15s %(message)s",
                    level=log.INFO if not config.getboolean('Inference', 'do_verbose') else log.DEBUG)


    visualizer = Visualizer(config)
    visualizer.run()


if __name__ == '__main__':
    main()
