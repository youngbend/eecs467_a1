import numpy as np, matplotlib as mpl, matplotlib.pyplot as plt
import imageio
import scipy.ndimage

def gradient(A, B):
    # Calculate an approximate gradient between two pixels values
    return A - B

class Target:
    def __init__(self, x, y):
        # Initialize tracker instance at center location (x,y)
        self.__center_x = x
        self.__center_y = y

    def get_center(self):
        return (self.__center_x, self.__center_y)

    def update(self, image):
        # Track target with image
        return (self.__center_x, self.__center_y)

class Tracker:
    def __init__(self, scan_offset, scan_border, target_offset, threshhold):
        # Initialize tracker with scanning offset (row_offset, col_offset),
        # scanning border (row_border, col_border), target bound offset,
        # and gradient threshhold
        self.__targets = []
        self.__scan_offset = scan_offset
        self.__border = scan_border
        self.__target_offset = target_offset
        self.__threshhold = threshhold

    def get_target_centers(self):
        return [t.get_center() for t in self.__targets]

    def scan(self, image):
        # Scan image for target. Scans horizontally from top to bottom
        for r in range(self.__border[0], image.shape[0] - self.__border[0], self.__scan_offset[0]):
            for c in range(self.__border[1], image.shape[1] - self.__border[1], self.__scan_offset[1]):
                if gradient(image[r,c-self.__scan_offset[1]], image[r,c]) > self.__threshhold:
                    try:
                        self.__rgb[r,c] = [255,0,0]
                    except:
                        continue

                    if self.pinpoint_target(image, r, c):
                        return

    def update_targets(self, image):
        for t in self.__targets:
            t.update(image)

    def pinpoint_target(self, image, row, col):
        # Pinpoint the target center given the triggering index
        top = None
        bottom = None
        left_right = [col, col]
        for r in range(row - self.__target_offset, image.shape[0]):
            edge_trigger = [False, False]
            for c in range(left_right[0] - self.__target_offset, left_right[1] + self.__target_offset):
                if gradient(image[r,c-1], image[r,c]) > self.__threshhold:
                    if not edge_trigger[0]:
                        edge_trigger[0] = True
                        left_right[0] = c
                        if not top:
                            top = r
                    elif not edge_trigger[1]:
                        edge_trigger[1] = True
                        left_right[1] = c
                        if not top:
                            top = r


        left = None
        right = None

        # center = (top + bottom) / 2

        # 1. find horizontal bar - start from center -> move left within radius
        # 2. if no bar found within the radius, immediately return False

        # image[top,:] = [255,0,0]
        # image[bottom,:] = [255,0,0]
        return True

    def attach_rgb(self, rgb):
        # For debugging purposes
        self.__rgb = rgb


tracker = Tracker((2,2), (10,10), 5, 40)

im = np.array(imageio.imread("tracking_dark.jpg"))
im = im[::5,::5]
# im = im[100:300,100:300]

gray = im.mean(2)

plt.figure()
plt.title('Image')
plt.axis('off')
plt.imshow(gray, cmap='gray')

tracker.attach_rgb(im)
tracker.scan(gray)

plt.figure()
plt.title('High contrast')
plt.axis('off')
plt.imshow(im)
plt.imsave("points_of_interest.jpg", im)
