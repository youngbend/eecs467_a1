import numpy as np
import math
import time
from picamera import PiCamera
from picamera.array import PiRGBArray
import pygame
import cv2
import tracker


tracker = tracker.Tracker((4,4), 5, 35, 15, 20)
camera_resolution = [640,480]
camera_framerate = 15
scan_period = 80
update_period = 1
dot_size = 7

pygame.init()
pygame.display.set_caption("Tracking Feed")
screen = pygame.display.set_mode(camera_resolution)
camera = PiCamera()
camera.resolution = tuple(camera_resolution)
camera.framerate = camera_framerate
rawCapture = PiRGBArray(camera, size=tuple(camera_resolution))
time.sleep(0.5)

frame_counter = 0

for frame in camera.capture_continuous(rawCapture, format="rgb", use_video_port=True):
    image = frame.array

    gray = image.mean(2)
    if frame_counter % scan_period == 0:
        tracker.scan(gray, (20,20,20,20))
    elif frame_counter % update_period == 0 and len(tracker.get_target_centers()) > 0:
        tracker.update_targets(gray)

    for center in tracker.get_target_centers():
        image = cv2.circle(image, (center[1], center[0]), dot_size, (255,0,0), -1)

    screen.fill([0,0,0])
    image = image.swapaxes(0,1)
    image = pygame.surfarray.make_surface(image)
    screen.blit(image, (0,0))
    pygame.display.update()

    rawCapture.truncate(0)
    frame_counter += 1

