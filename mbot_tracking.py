import numpy as np
from picamera import PiCamera
from picamera.array import PiRGBArray
import pygame
import time
import sys
import cv2
import tracker
import lcm
from lcmtypes.simple_motor_command_t import simple_motor_command_t

tracker = tracker.Tracker((4,4), 5, 35, 20, 15)
camera_resolution = [640,480]
camera_framerate = 15
scan_period = 50
update_period = 1
dot_size = 7

FORWARD_VEL_CONST = 0.3
ANGULAR_VEL_CONST = 0.15

lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=1")
pygame.init()
pygame.display.set_caption("Tracking Feed")
screen = pygame.display.set_mode(camera_resolution)
camera = PiCamera()
camera.resolution = tuple(camera_resolution)
camera.framerate = camera_framerate
rawCapture = PiRGBArray(camera, size=tuple(camera_resolution))

frame_counter = 0

for frame in camera.capture_continuous(rawCapture, format="rgb", use_video_port=True):
    image = frame.array

    gray = image.mean(2)
    if frame_counter % scan_period == 0:
        tracker.scan(gray, (20,20,20,20))
        image = cv2.putText(image, 'Scan', (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0))
    elif frame_counter % update_period == 0 and len(tracker.get_target_centers()) > 0:
        image = cv2.putText(image, 'Track...', (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255))
        tracker.update_targets(gray)

    for center in tracker.get_target_centers():
        image = cv2.circle(image, (center[1], center[0]), dot_size, (255,0,0), -1)

    screen.fill([0,0,0])
    image = image.swapaxes(0,1)
    image = pygame.surfarray.make_surface(image)
    screen.blit(image, (0,0))
    pygame.display.update()

    command = simple_motor_command_t()
    command.utime = int(time.time() * 1000000)
    command.angular_velocity = 0
    command.forward_velocity = 0

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
            cv2.destroyAllWindows()

    key_input = pygame.key.get_pressed()
    if key_input[pygame.K_LEFT]:
        command.angular_velocity += ANGULAR_VEL_CONST
    if key_input[pygame.K_UP]:
        command.forward_velocity += FORWARD_VEL_CONST
    if key_input[pygame.K_RIGHT]:
        command.angular_velocity -= ANGULAR_VEL_CONST
    if key_input[pygame.K_DOWN]:
        command.forward_velocity -= FORWARD_VEL_CONST
    if key_input[pygame.K_q]:
        pygame.quit()
        sys.exit()
        cv2.destroyAllWindows()

    lc.publish("MBOT_MOTOR_COMMAND_SIMPLE", command.encode())

    rawCapture.truncate(0)
    frame_counter += 1
