import numpy as np
from picamera import PiCamera
from picamera.array import PiRGBArray
import pygame
import sys
import cv2
import tracker
import lcm
from lcmtypes import mbot_motor_pwm_t

tracker = tracker.Tracker((4,4), 5, 35, 15, 30)
camera_resolution = [640,480]
camera_framerate = 15
scan_period = 80
update_period = 1
dot_size = 7

FWD_PWM_CMD = 0.3
TURN_PWM_CMD = 0.3

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
        image = cv2.putText(image, 'Scan', (20,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))
    elif frame_counter % update_period == 0 and len(tracker.get_target_centers()) > 0:
        image = cv2.putText(image, 'Track...', (20,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))
        tracker.update_targets(gray)

    for center in tracker.get_target_centers():
        image = cv2.circle(image, (center[1], center[0]), dot_size, (255,0,0), -1)

    screen.fill([0,0,0])
    image = image.swapaxes(0,1)
    image = pygame.surfarray.make_surface(image)
    screen.blit(image, (0,0))
    pygame.display.update()

    fwd = 0.0
    turn = 0.0

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
            cv2.destroyAllWindows()

    key_input = pygame.key.get_pressed()
    if key_input[pygame.K_LEFT]:
        turn += 1.0
    if key_input[pygame.K_UP]:
        fwd += 1.0
    if key_input[pygame.K_RIGHT]:
        turn -= 1.0
    if key_input[pygame.K_DOWN]:
        fwd -= 1.0
    if key_input[pygame.K_q]:
        pygame.quit()
        sys.exit()
        cv2.destroyAllWindows()

    command = mbot_motor_pwm_t()
    command.left_motor_pwm =  fwd * FWD_PWM_CMD - turn * TURN_PWM_CMD
    command.right_motor_pwm = fwd * FWD_PWM_CMD + turn * TURN_PWM_CMD
    lc.publish("MBOT_MOTOR_PWM",command.encode())

    rawCapture.truncate(0)
    frame_counter += 1
