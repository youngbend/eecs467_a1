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

    def update(self, new_center_x, new_center_y):
        # Track target with image
        self.__center_x = new_center_x
        self.__center_y = new_center_y
        return (self.__center_x, self.__center_y)

class Tracker:
    def __init__(self, scan_offset, scan_border, target_offset, threshhold):
        # Initialize tracker with scanning offset (row_offset, col_offset),
        # scanning border (row_border, col_border), target bound offset,
        # and gradient threshhold

        self.__targets = []

        # (row_offset, col_offset) responsible for determining the distance
        # between samples within the global search.
        self.__scan_offset = scan_offset

        # (row_border, col_border) giving a margin around the image that is not
        # searched (due to distortion of the fisheye lens)
        self.__border = scan_border

        # offset around localized feature search that determines how far outside
        # the bounds of the previous layer we should search
        self.__target_offset = target_offset

        # Threshhold for what we consider to be a significant gradient
        self.__threshhold = threshhold

    def get_target_centers(self):
        # Return the center of all tracked targets
        return [t.get_center() for t in self.__targets]

    def scan(self, image):
        # Scan image for target. Scans horizontally from top to bottom
        for r in range(self.__border[0], image.shape[0] - self.__border[0], self.__scan_offset[0]):
            for c in range(self.__border[1], image.shape[1] - self.__border[1], self.__scan_offset[1]):

                # If we find a significant rising gradient (left side of the cross),
                # start a localized search to distinguish features from false positives
                if gradient(image[r,c-self.__scan_offset[1]], image[r,c]) > self.__threshhold:

                    # (DEBUG) Turn the significant pixel red in the image
                    # try:
                    #     self.__rgb[r,c] = [255,0,0]
                    # except:
                    #     None

                    # If the localized search finds a target, stop the global search
                    # Qingyi: assumes that the pinpoint_traget returns the tuple:
                    # (Boolean:whether there is a cross, center.x, center.y)
                    is_cross, cross_x, cross_y = self.pinpoint_target(image, r, c)
                    if is_cross: 
                        self.__targets.append(Target(cross_x, cross_y))
                    # Qingyi: We would need to avoid adding a target twice though - 
                    # this can be done by checking r and c is not in the range of 
                    # existing cross region
        return 
        

    def update_targets(self, image):
        # For task 3
        for t in self.__targets:
            t.update(image)

    def pinpoint_target(self, image, row, col):
        # Pinpoint the target center given the triggering index

        # Define a smaller scan offset for the localized search so we can detect
        # finer features
        scan_offset = [int(self.__scan_offset[0] / 2), int(self.__scan_offset[1] / 2)]

        # Column numbers of the vertical bar top and bottom in the order:
        # [top_left, top_right, bottom_left, bottom_right]
        vbar_bounds = [None, None, None, None]

        # Row numbers of top and bottom
        top = None
        bottom = None

        # Keep track of the minimum intensity so we know what intensity black is
        # regardless of color distortion
        min_intensity = 255

        # Bar width of previous row which is updated to track the bar, even when angled.
        # Search starts at the column of the detected feature with offset on
        # both sides.
        left_right = [col - self.__target_offset, col + self.__target_offset]

        # Iterate through the rows until the bottom of the image.
        for r in range(row - self.__target_offset, image.shape[0], scan_offset[0]):

            # If our search goes outside of the boundaries, return.
            if r < self.__border[0] or r > (image.shape[0] - self.__border[0]):
                return False

            # Per row, keep track of whether the left edge was triggered
            edge_trigger = False

            # Keep track whether either gradient was triggered so we know if we
            # are still in the bounds of the object
            cross_encounter = False

            # Keep track of the minimum row intensity to know whether the row
            # contained a black pixel or not.
            min_row_intensity = 255

            # Iterate between the position of the previous layer, accounting for
            # diagonal lines with the offset.
            for c in range(left_right[0] - self.__target_offset, left_right[1] + self.__target_offset, scan_offset[1]):

                # If the horizontal bar goes outside the boundaries, return
                if c < self.__border[1] or c > (image.shape[1] - self.__border[1]):
                    return False

                # Encounter left edge on first rising gradient
                if gradient(image[r,c-scan_offset[1]], image[r,c]) > self.__threshhold:
                    if not edge_trigger:
                        edge_trigger = True
                        cross_encounter = True
                        left_right[0] = c

                        # (DEBUG) Color rising edge blue
                        try:
                            self.__rgb[r,c] = [0,0,255]
                        except:
                            None

                        # Define top for first edge hit
                        if not top:
                            top = r
                            vbar_bounds[0] = c

                # Encountered right edge on falling gradient after left edge
                elif gradient(image[r,c-1], image[r,c]) < -self.__threshhold:
                    if edge_trigger:
                        left_right[1] = c
                        cross_encounter = True

                        # (DEBUG) Color falling edge green
                        try:
                            self.__rgb[r,c] = [0,255,0]
                        except:
                            None

                        # Mark the top right corner of the bar
                        if top and not vbar_bounds[1]:
                            vbar_bounds[1] = c

                        continue

                # For checking if we encountered the cross within the row
                if top and image[r,c] < min_row_intensity:
                    min_row_intensity = image[r,c]
                    if min_row_intensity < min_intensity:
                        min_intensity = min_row_intensity

            # If a black pixel was not encountered in this row, mark the bottom
            if top and not ((min_row_intensity - min_intensity) < self.__threshhold) and not cross_encounter:
                bottom = r - scan_offset[0]
                vbar_bounds[2] = left_right[0]
                vbar_bounds[3] = left_right[1]
                break

        # If all of the features of the vertical bar were not detected, the feature
        # is probably not a cross
        if not top or not bottom or not all(vbar_bounds):
            return False

        # (DEBUG)
        print(vbar_bounds)

        # (DEBUG) Draw red bars on the top and bottom of the vertical bar
        try:
            self.__rgb[top,vbar_bounds[0]:vbar_bounds[1]] = [255,0,0]
            self.__rgb[bottom,vbar_bounds[2]:vbar_bounds[3]] = [255,0,0]
        except:
            None

        left = None
        right = None

        center = (top + bottom) / 2

        # 1. find horizontal bar - start from center -> move left within radius
        # 2. if no bar found within the radius, immediately return False

        return True

    def attach_rgb(self, rgb):
        # Link the RGB to the class object so we can draw on it.
        self.__rgb = rgb


tracker = Tracker((2,2), (10,10), 2, 35)

im = np.array(imageio.imread("tracking_dark.jpg"))

# Downsample the image to the (approximate) MBot resolution
im = im[::5,::5]

im = im[100:300,100:300]
# im = im[600:1000,800:1500]


gray = im.mean(2)

# Probably not necessary, but use a sharpening filter to alias the cross
# gray = gray - scipy.ndimage.gaussian_filter(gray, 50)

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
