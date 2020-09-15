import numpy as np, matplotlib as mpl, matplotlib.pyplot as plt
import math
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
                    # try:
                    #     self.__rgb[r,c] = [255,0,0]
                    # except:
                    #     None

                    if self.pinpoint_target(image, r, c):
                        return

    def update_targets(self, image):
        for t in self.__targets:
            t.update(image)

    def pinpoint_target(self, image, row, col):
        # Pinpoint the target center given the triggering index

        # Smaller scan offset for the localized search
        scan_offset = [int(self.__scan_offset[0] / 2), int(self.__scan_offset[1] / 2)]

        # Column numbers of the vertical bar top and bottom in the order:
        # [top_left, top_right, bottom_left, bottom_right]
        vbar_bounds = [None, None, None, None]

        # Row numbers of top and bottom
        top = None
        bottom = None

        min_intensity = 255

        # Bar width of previous row which is update to track the bar
        left_right = [col - self.__target_offset, col + self.__target_offset]

        for r in range(row - self.__target_offset, image.shape[0], scan_offset[0]):

            if r < self.__border[0] or r > (image.shape[0] - self.__border[0]):
                return False

            edge_trigger = False
            cross_encounter = False
            min_row_intensity = 255
            for c in range(left_right[0] - self.__target_offset, left_right[1] + self.__target_offset, scan_offset[1]):

                if c < self.__border[1] or c > (image.shape[1] - self.__border[1]):
                    return False
                # Encounter left edge on first rising gradient
                if gradient(image[r,c-scan_offset[1]], image[r,c]) > self.__threshhold:
                    if not edge_trigger:
                        edge_trigger = True
                        cross_encounter = True
                        left_right[0] = c

                        # Color rising edge blue
                        self.__rgb[r,c] = [0,0,255]

                        # Define top for first edge hit
                        if not top:
                            top = r
                            vbar_bounds[0] = c

                # Encountered right edge on falling gradient after left edge
                elif gradient(image[r,c-1], image[r,c]) < -self.__threshhold:
                    if edge_trigger:
                        left_right[1] = c
                        cross_encounter = True

                        # Color falling edge green
                        self.__rgb[r,c] = [0,255,0]

                        # Mark the top right corner of the bar
                        if top and not vbar_bounds[1]:
                            vbar_bounds[1] = c

                        continue

                # For checking if we encountered the cross within the row
                if top and image[r,c] < min_row_intensity:
                    min_row_intensity = image[r,c]
                    if min_row_intensity < min_intensity:
                        min_intensity = min_row_intensity

            # If a black pixel was not encountered in this row
            if top and not ((min_row_intensity - min_intensity) < self.__threshhold) and not cross_encounter:
                bottom = r - scan_offset[0]
                vbar_bounds[2] = left_right[0]
                vbar_bounds[3] = left_right[1]
                break

        print(vbar_bounds)

        if not top or not bottom or not all(vbar_bounds):
            return False

        # Draw bars on the top and bottom of the vertical bar
        try:
            self.__rgb[top,vbar_bounds[0]:vbar_bounds[1]] = [255,0,0]
            self.__rgb[bottom,vbar_bounds[2]:vbar_bounds[3]] = [255,0,0]
        except:
            None

        left = None
        right = None

        # center = (top + bottom) / 2

        # 1. find horizontal bar - start from center -> move left within radius
        # 2. if no bar found within the radius, immediately return False

        # self.__rgb[top,:] = [255,0,0]
        # self.__rgb[bottom,:] = [255,0,0]
        # print(vbar_bounds)
        # try:
        #     self.__rgb[top,:] = [255,0,0]
        #     self.__rgb[:,vbar_bounds[0] = [0,0,255]
        #     self.__rgb[:,vbar_bounds[1] = [0,255,0]
        # except:
        #     None

        return True

    def attach_rgb(self, rgb):
        # For debugging purposes
        self.__rgb = rgb


tracker = Tracker((2,2), (10,10), 2, 15)

im = np.array(imageio.imread("tracking_dark.jpg"))
im = im[::5,::5]
im = im[100:300,100:300]
# im = im[600:1000,800:1500]

gray = im.mean(2)
print(gray.shape)
gray = gray - scipy.ndimage.gaussian_filter(gray, 90)
print(math.floor(gray.shape[0]*0.04))

plt.figure()
plt.title('Image')
plt.axis('off')
plt.imshow(gray, cmap='gray')
plt.imsave("grayscale.jpg", gray, cmap='gray')

tracker.attach_rgb(im)
tracker.scan(gray)

plt.figure()
plt.title('Tracker output')
plt.axis('off')
plt.imshow(im)
plt.imsave("points_of_interest.jpg", im)
