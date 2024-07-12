import numpy as np
import cv2 as cv

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    for _, landmark in enumerate(landmarks.landmark):#index is ignored that is why it is replaced with _
        landmark_x = min(int(landmark.x * image_width), image_width - 1)#landmark.x and landmark.y: Normalized coordinates of the landmark (values between 0 and 1).
        landmark_y = min(int(landmark.y * image_height), image_height - 1)#Ensures that the y-coordinate does not exceed the image height.
        landmark_point.append([landmark_x, landmark_y])
        
    return landmark_point
