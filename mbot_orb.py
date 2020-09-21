import pygame
from pygame.locals import *
from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import time
import numpy as np
import sys
sys.path.append("lcmtypes")
import lcm
from lcmtypes import mbot_motor_pwm_t
from lcmtypes import simple_motor_command_t
flip_h = 0
flip_v = 0

lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=1")
pygame.init()
pygame.display.set_caption("ORB Feature Tracking")
screen = pygame.display.set_mode([640,480])
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))
time.sleep(0.5)
last_keypoints = []
last_descriptors = []

for frame in camera.capture_continuous(rawCapture, format="rgb", use_video_port=True):
    image = frame.array
    
    fwd = 0.0
    turn = 0.0
    for event in pygame.event.get():
        if event.type==pygame.QUIT:
            pygame.quit()
            sys.exit()
            cv2.destroyAllWindows()
    key_input = pygame.key.get_pressed()  
    if key_input[pygame.K_LEFT]:
        turn += 1.0
    if key_input[pygame.K_UP]:
        fwd +=1.0
    if key_input[pygame.K_RIGHT]:
        turn -= 1.0
    if key_input[pygame.K_DOWN]:
        fwd -= 1.0
    if key_input[pygame.K_h]:
        if flip_h == 0:
            flip_h = 1
        else:
            flip_h = 0
    if key_input[pygame.K_v]:
        if flip_v == 0:
            flip_v = 1
        else:
            flip_v = 0
    if key_input[pygame.K_q]:
            pygame.quit()
            sys.exit()
            cv2.destroyAllWindows()
            
    command = simple_motor_command_t.simple_motor_command_t()
    command.forward_velocity = fwd
    command.angular_velocity = turn
    lc.publish("MBOT_MOTOR_COMMAND_SIMPLE",command.encode())



    # Start ORB part
    im = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create()

    # find the keypoints and compute their descriptors
    keypoints, descriptors = orb.detectAndCompute(im, None)
    if len(last_keypoints) == 0:
        #draw keypoints
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
        cv2.drawKeypoints(im, keypoints, im, color=(0,255,0))
    else:
        # Create a Brute Force Matcher object.
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

        # Perform the matching between the ORB descriptors of the training image and the test image
        matches = bf.match(descriptors, last_descriptors)
        matched_keypoint = [keypoints[i] for i in [match.queryIdx for match in matches]]
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
        cv2.drawKeypoints(im, keypoints, im, color=(0,255,0))

    last_keypoints = keypoints
    last_descriptors = descriptors


    screen.fill([0,0,0])
    
    im = im.swapaxes(0,1)
    #image = cv2.flip(image, -1)
    im = pygame.surfarray.make_surface(im)
    screen.blit(im, (0,0))
    pygame.display.update()
    rawCapture.truncate(0)
