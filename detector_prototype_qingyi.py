import cv2
import numpy as np
#np.set_printoptions(threshold=np.inf)

im = cv2.imread("cross_image.jpg")
#cv2.imshow("Image", im)
#cv2.waitKey(0)
im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

#im = im[225:280, 320:380]
#print(im)
#cv2.imshow("Image", im)
#cv2.waitKey(0)



left_boundary = -1
right_boundary = -1

left_boundaries = []

def find_cross_boundary(im):
    guess_left_boundaries_uv = []
    guess_right_boundaries_uv = []
    for i in range(im.shape[0]):
        guess_left_boundary = (-1,-1)
        guess_right_boundary = (-1,-1)
        for j in range(im.shape[1] - 2):
            if (im[i,j+1] <= 60 or int(im[i,j+1]) + int(im[i,j+2]) <= 120) and im[i,j] >= 80 and int(im[i,j]) - int(im[i,j+1]) > 40:
                #print("At", i, j)
                #print("Left:", im[i,j])
                #print("Right:", im[i,j+1])
                #guess_left_boundaries.append((j,i))
                guess_left_boundary = (i,j)
            if (im[i,j] <= 60 or int(im[i,j+1]) + int(im[i,j+2]) >= 200) and im[i,j+1] >= 80 and int(im[i,j]) - int(im[i,j+1]) < -40:
                guess_right_boundary = (i,j)
        if guess_left_boundary[0] != -1 and guess_right_boundary[0] != -1 and 0 < guess_right_boundary[1] - guess_left_boundary[1] < 100:
            guess_left_boundaries_uv.append((guess_left_boundary[1], guess_left_boundary[0]))
            guess_right_boundaries_uv.append((guess_right_boundary[1], guess_right_boundary[0]))    
    
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    for point in guess_left_boundaries_uv:
        im = cv2.circle(im, point, 1, (0, 0, 255))
    for point in guess_right_boundaries_uv:
        im = cv2.circle(im, point, 1, (255, 0, 0))
    return im

im = find_cross_boundary(im)
cv2.imshow("Image", im)

cv2.waitKey(0)
cv2.destroyAllWindows()