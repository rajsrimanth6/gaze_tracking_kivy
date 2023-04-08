# Import kivy dependencies first
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

# Import kivy UX components
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label

# Import other kivy stuff
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger


import math
import mediapipe as mp
import numpy as np
import cv2 as cv
import time
from time import sleep


# variables
global both_counter
global both_final_counter
global counter
global final_counter
global right_counter
global right_final_counter
global blink_time
global blink_list
global count
global check
check = 1

both_counter = 0
both_final_counter = 0
counter = 0
final_counter = 0
right_counter = 0
right_final_counter = 0
blink_time = [0, 0]
blink_list = []
count = 0
mp_face_mesh = mp.solutions.face_mesh
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390,
            249, 263, 466, 388, 387, 386, 385, 384, 398]

RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154,
             155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_IRIS = [469, 470, 471, 472]
RIGHT_IRIS = [474, 475, 476, 477]

# L_H_LEFT = [33]  # right eye right most landmark
# L_H_RIGHT = [133]  # right eye left most landmark
# R_H_LEFT = [362]  # left eye right most landmark
# R_H_RIGHT = [263]  # left

# R_UP = [159]
# R_DOWN = [145]
L_H_LEFT = 33  # right eye right most landmark
L_H_RIGHT = 133  # right eye left most landmark
R_H_LEFT = 362  # left eye right most landmark
R_H_RIGHT = 263  # left

#
L_UP = 386  # left eye uppermost  coordinate
L_DOWN = 374  # left eye downmost  coordinate


R_UP = 159  # right eye uppermost  coordinate
R_DOWN = 145  # right eye downmost  coordinate
landmarks = []


class CamApp(App):

    def build(self):
        # Main layout components
        self.web_cam = Image(height=25, width=35)
        self.button1 = Button(
            text="close", on_press=self.stop, size_hint=(1, .1))
        self.button2 = Button(
            text="open", on_press=self.start, size_hint=(1, .1))
        self.mainLayout = BoxLayout(orientation='vertical')
        self.buttonLayout = BoxLayout(orientation='vertical')
        self.camLayout = BoxLayout(orientation='vertical')
        self.label = Label(text='Original text', size_hint=(1, .1))
        # Add items to layout
        # self.mainlayout = BoxLayout(orientation='vertical')
        self.camLayout.add_widget(self.web_cam)
        self.buttonLayout.add_widget(self.button1)
        self.buttonLayout.add_widget(self.button2)
        self.mainLayout.add_widget(self.camLayout)
        self.mainLayout.add_widget(self.buttonLayout)
        # self.buttonLayout.add_widget(self.iris_position_label)
        self.buttonLayout.add_widget(self.label)

        # Setup video capture device
        # self.capture = cv.VideoCapture(0)
        # Clock.schedule_interval(self.update, 1.0/33.0)
        self.setup()

        return self.mainLayout

    def setup(self):
        self.capture = cv.VideoCapture(0)
        # self.capture.get(cv.CAP_PROP_FPS))
        Clock.schedule_interval(self.update, 1/25.0)

    def update(self, *args):
        global check
        # Read frame from opencv
        ret, self.frame = self.capture.read()
        if not ret:
            if check == 1:
                print("closed")
                check = 0
            pass
        else:
            self.frame = cv.flip(self.frame, 1)
            # frame = frame[120:120+250, 200:200+250, :]

            # Flip horizontall and convert image to texture
            # algorithm
            with mp_face_mesh.FaceMesh(max_num_faces=1,
                                       refine_landmarks=True,
                                       min_detection_confidence=0.5,
                                       min_tracking_confidence=0.5
                                       ) as face_mesh:
                rgb_frame = cv.cvtColor(self.frame, cv.COLOR_BGR2RGB)
                img_h, img_w = self.frame.shape[:2]
                results = face_mesh.process(rgb_frame)
                if results.multi_face_landmarks:
                    mesh_points0 = np.array([tuple(np.multiply([p.x, p.y], [img_w, img_h]).astype(
                        int).ravel()) for p in results.multi_face_landmarks[0].landmark])
                    mesh_points = [tuple(np.multiply([p.x, p.y], [img_w, img_h]).astype(
                        int).ravel()) for p in results.multi_face_landmarks[0].landmark]
                    # print(results.multi_face_landmarks)
                    # print(mesh_points)

                    (l_cx, l_cy), l_radius = cv.minEnclosingCircle(
                        mesh_points0[LEFT_IRIS])
                    (r_cx, r_cy), r_radius = cv.minEnclosingCircle(
                        mesh_points0[RIGHT_IRIS])

                    center_left = tuple(np.array([l_cx, l_cy], dtype=np.int32))
                    center_right = tuple(
                        np.array([r_cx, r_cy], dtype=np.int32))

                    # print(mesh_points[R_UP], mesh_points[R_DOWN],mesh_points[R_H_RIGHT], mesh_points[R_H_LEFT])

                    cv.circle(self.frame, center_left, int(l_radius),
                              (255, 0, 255), 1, cv.LINE_AA)
                    cv.circle(self.frame, center_right, int(r_radius),
                              (255, 0, 255), 1, cv.LINE_AA)
                    self.iris_pos, ratio = self.iris_position(
                        center_right, mesh_points[R_UP], mesh_points[R_DOWN], mesh_points[R_H_RIGHT], mesh_points[R_H_LEFT])
                    print(self.iris_pos)

                    self.label.text = self.iris_pos

                    cv.putText(self.frame, f"Iris pos: {self.iris_pos}", (
                        30, 30), cv.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 1, cv.LINE_AA)
                    buf = cv.flip(self.frame, 0).tostring()
                    img_texture = Texture.create(
                        size=(self.frame.shape[1], self.frame.shape[0]), colorfmt='bgr')
                    img_texture.blit_buffer(
                        buf, colorfmt='bgr', bufferfmt='ubyte')
                    self.web_cam.texture = img_texture

    def stop(self, arg):
        self.loop = True
        self.camLayout.remove_widget(self.web_cam)
        self.capture.release()

    def start(self, arg):
        # exit.set()
        self.camLayout.add_widget(self.web_cam)
        self.setup()

    def iris_position(self, iris_center, up_point, down_point, right_point, left_point):
        center_to_right_dist = self.euclidean_distance(
            iris_center, right_point)
        total_distance = self.euclidean_distance(right_point, left_point)

        ratio = center_to_right_dist/total_distance

        diff = down_point[1]-up_point[1]
        if diff < 9:
            iris_position = "down"
        elif diff > 11:
            iris_position = "up"
        elif ratio > 0.42 and ratio <= 0.57:
            iris_position = "center"
        elif ratio <= 0.42:
            iris_position = "right"
        else:
            iris_position = "left"
        ratio_vertical = diff
        return iris_position, ratio_vertical

    def euclidean_distance(self, point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        self.distance = math.sqrt((x2 - x1)**2 + (y2-y1)**2)
        return self.distance


if __name__ == '__main__':
    CamApp().run()
