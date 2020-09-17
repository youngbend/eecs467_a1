import numpy as np, matplotlib as mpl, matplotlib.pyplot as plt
import math
import imageio
import scipy.ndimage
import cv2

def gradient(A, B):
    # Calculate an approximate gradient between two pixels values
    return int(A) - int(B)

class Target:
    def __init__(self, x, y, radius):
        # Initialize tracker instance at center location (x,y)
        self.__is_active = True
        self.__center_x = x
        self.__center_y = y
        self.__radius = radius
        self.__history = [(x,y)]
        # Qingyi: add a radius of the target, to avoid looking at that region agin
        # when looking for new targets; add history

    def get_center(self):
        return (self.__center_x, self.__center_y)

    def get_radius(self):
        return self.__radius

    def update(self, image):
        # Track target with image
        print("Running update on current Trackers...")
        print("For the tracker at ",self.__center_x, self.__center_y, "with radius", self.__radius, "...")
        is_found, x, y, r = Tracker((2,2),(10,10),self.__radius + 15,35).pinpoint_target(image, self.__center_x, self.__center_y)
        if is_found:
            self.__center_x = x
            self.__center_y = y
            self.__radius = r + 5 
            self.__history.append((x,y))
            print("Updated Successful!")
        else:
            self.__is_active = False
            print("Not Found any more.")
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
        self.update_targets(image)

        for r in range(self.__border[0], image.shape[0] - self.__border[0], self.__scan_offset[0]):
            for c in range(self.__border[1], image.shape[1] - self.__border[1], self.__scan_offset[1]):
                for t in self.__targets:
                    #Qingyi: skip the region that has already had a target
                    x, y = t.get_center()
                    radius = t.get_radius()
                    if x - radius < r < x + radius and y - radius < r < y + radius:
                        continue

                # If we find a significant rising gradient (left side of the cross),
                # start a localized search to distinguish features from false positives
                if gradient(image[r,c-self.__scan_offset[1]], image[r,c]) > self.__threshhold:

                    # (DEBUG) Turn the significant pixel red in the image
                    # try:
                    #     self.__rgb[r,c] = [255,0,0]
                    # except:
                    #     None

                    # If the localized search finds a target, stop the global search
                    is_target, center_row, center_column, radius = self.pinpoint_target(image, r, c)
                    if is_target:
                        self.__targets.append(Target(center_row, center_column, radius))
                        print("New Target Found")
                        return

    def update_targets(self, image):
        # For task 3
        for t in self.__targets:
            t.update(image)

    def pinpoint_target(self, image, row, col):
        # Pinpoint the target center given the triggering index

        # Define a smaller scan offset for the localized search so we can detect
        # finer features
        scan_offset = [math.ceil(self.__scan_offset[0] / 2), math.ceil(self.__scan_offset[1] / 2)]

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
                return (False, 0, 0, 0)

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
                    return (False, 0, 0, 0)

                # Encounter left edge on first rising gradient
                if gradient(image[r,c-scan_offset[1]], image[r,c]) > self.__threshhold:
                    cross_encounter = True
                    if not edge_trigger:
                        edge_trigger = True
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
                elif gradient(image[r,c-scan_offset[1]], image[r,c]) < -self.__threshhold:
                    cross_encounter = True
                    if edge_trigger:
                        left_right[1] = c

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
            return (False, 0, 0, 0)

        # If the top and bottom were not similar in width
        if abs((vbar_bounds[1] - vbar_bounds[0]) - (vbar_bounds[3] - vbar_bounds[2])) > self.__target_offset:
            return (False, 0, 0, 0)

        left = None
        right = None

        center_row = (top + bottom) // 2
        center_column = int(np.mean(vbar_bounds))
        column_radius = (vbar_bounds[1] - vbar_bounds[0]) // 2

        # [top_left, bottom_left, top_right, bottom_right]
        hbar_bounds = [None, None, None, None]

        # Perform a vertical scan from the center until we find the left edge
        up_down = [center_row - column_radius, center_row + column_radius]
        for c in range(center_column, 0, -scan_offset[1]):

            if c < self.__border[1]:
                return (False, 0, 0, 0)

            edge_trigger = False
            cross_encounter = False
            min_col_intensity = 255

            for r in range(up_down[0] - self.__target_offset, up_down[1] + self.__target_offset, scan_offset[0]):

                if r < self.__border[0] or r > (image.shape[0] - self.__border[0]):
                    return (False, 0, 0, 0)

                if gradient(image[r-scan_offset[0],c], image[r,c]) > self.__threshhold:
                    cross_encounter = True
                    if not edge_trigger:
                        edge_trigger = True
                        up_down[0] = r

                        # (DEBUG) Color rising edge blue
                        try:
                            self.__rgb[r,c] = [0,0,255]
                        except:
                            None

                elif gradient(image[r-scan_offset[0],c], image[r,c]) < -self.__threshhold:
                    cross_encounter = True
                    if edge_trigger:
                        up_down[1] = r

                        # (DEBUG) Color falling edge green
                        try:
                            self.__rgb[r,c] = [0,255,0]
                        except:
                            None

                        continue

                if image[r,c] < min_col_intensity:
                    min_col_intensity = image[r,c]

            if not cross_encounter and abs(int(min_col_intensity) - int(min_intensity)) > self.__threshhold:
                left = c
                hbar_bounds[0] = up_down[0]
                hbar_bounds[1] = up_down[1]
                break

        # Perform a vertical scan from the center until we find the right edge
        up_down = [center_row - column_radius, center_row + column_radius]
        for c in range(center_column, image.shape[1], scan_offset[1]):

            if c > (image.shape[1] - self.__border[1]):
                return (False, 0, 0, 0)

            edge_trigger = False
            cross_encounter = False
            min_col_intensity = 255

            for r in range(up_down[0] - self.__target_offset, up_down[1] + self.__target_offset, scan_offset[0]):

                if r < self.__border[0] or r > (image.shape[0] - self.__border[0]):
                    return (False, 0, 0, 0)

                if gradient(image[r-scan_offset[0],c], image[r,c]) > self.__threshhold:
                    cross_encounter = True
                    if not edge_trigger:
                        edge_trigger = True
                        up_down[0] = r

                        # (DEBUG) Color rising edge blue
                        try:
                            self.__rgb[r,c] = [0,0,255]
                        except:
                            None

                elif gradient(image[r-scan_offset[0],c], image[r,c]) < -self.__threshhold:
                    cross_encounter = True
                    if edge_trigger:
                        up_down[1] = r

                        # (DEBUG) Color falling edge green
                        try:
                            self.__rgb[r,c] = [0,255,0]
                        except:
                            None

                        continue

                if image[r,c] < min_col_intensity:
                    min_col_intensity = image[r,c]

            if not cross_encounter and abs(int(min_col_intensity) - int(min_intensity)) > self.__threshhold:
                right = c
                hbar_bounds[2] = up_down[0]
                hbar_bounds[3] = up_down[1]
                break

        # If all bounds were not satisfied
        if not left or not right or not all(hbar_bounds):
            return (False, 0, 0, 0)

        # If the right and left sides were not close to the same width
        if abs((hbar_bounds[1] - hbar_bounds[0]) - (hbar_bounds[3] - hbar_bounds[2])) > self.__target_offset:
            return (False, 0, 0, 0)

        # Find true center column
        center_column = (right + left) // 2

        # Extract the vector cooresponding to the upwards vertical arm of the cross with
        # the origin at the center
        up_vec = np.array([(vbar_bounds[1] + vbar_bounds[0]) / 2 - center_column,  center_row - top])
        if np.all((up_vec == 0)):
            return (False, 0, 0, 0)
        up_unit = up_vec / np.linalg.norm(up_vec)

        # Extract downward vertical vector
        down_vec = np.array([(vbar_bounds[3] + vbar_bounds[2]) / 2 - center_column, center_row - bottom])
        if np.all((down_vec == 0)):
            return (False, 0, 0, 0)
        down_unit = down_vec / np.linalg.norm(down_vec)

        # Extract right horizontal vector
        right_vec = np.array([right - center_column, center_row - (hbar_bounds[3] + hbar_bounds[2]) / 2])
        if np.all((right_vec == 0)):
            return (False, 0, 0, 0)
        right_unit = right_vec / np.linalg.norm(right_vec)

        # Extract left horizontal vector
        left_vec = np.array([left - center_column, center_row - (hbar_bounds[1] + hbar_bounds[0]) / 2])
        if np.all((left_vec == 0)):
            return (False, 0, 0, 0)
        left_unit = left_vec / np.linalg.norm(left_vec)

        # (DEBUG)
        # print(up_vec)
        # print(down_vec)
        # print(right_vec)
        # print(left_vec)

        # Check that vectors are about the same length
        if abs(np.linalg.norm(up_vec) - np.linalg.norm(right_vec)) > self.__target_offset:
            return (False, 0, 0, 0)

        # Check that vectors are not tiny
        if np.linalg.norm(down_vec) < self.__target_offset or np.linalg.norm(left_vec) < self.__target_offset:
            return (False, 0, 0, 0)

        # Classify as not a cross if the angle is greater than 17 degrees
        if abs(np.arccos(np.dot(up_unit, right_unit)) - (np.pi/2)) > 0.3:
            return (False, 0, 0, 0)

        if abs(np.arccos(np.dot(right_unit, down_unit)) - (np.pi/2)) > 0.3:
            return (False, 0, 0, 0)

        if abs(np.arccos(np.dot(down_unit, left_unit)) - (np.pi/2)) > 0.3:
            return (False, 0, 0, 0)

        if abs(np.arccos(np.dot(left_unit, up_unit)) - (np.pi/2)) > 0.3:
            return (False, 0, 0, 0)

        # (DEBUG)
        print(center_row, center_column)

        # (DEBUG) Draw lines on the bounds
        try:
            self.__rgb[top,vbar_bounds[0]:vbar_bounds[1]] = [255,0,0]
            self.__rgb[bottom,vbar_bounds[2]:vbar_bounds[3]] = [255,0,0]
            self.__rgb[hbar_bounds[0]:hbar_bounds[1], left] = [255,0,0]
            self.__rgb[hbar_bounds[2]:hbar_bounds[3], right] = [255,0,0]
        except:
            None

        # (DEBUG) Draw dot on the center
        try:
            self.__rgb[center_row-column_radius:center_row+column_radius, center_column-column_radius:center_column+column_radius] = [255,0,0]
        except:
            None

        return (True, center_row, center_column, max(bottom-top, right-left) // 2) # Qingyi: add the last item as the radius

    def attach_rgb(self, rgb):
        # Link the RGB to the class object so we can draw on it.
        self.__rgb = rgb


tracker = Tracker((2,2), (10,10), 5, 35)



for i in range(0,3):
    im = cv2.imread("image"+ str(i) + ".jpg")
    im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    tracker.scan(im)
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    for target_center in tracker.get_target_centers():
        im = cv2.circle(im, (target_center[1], target_center[0]), 2, (0, 0, 255),5)
    
    cv2.imshow("Tracked Image" + str(i), im)
    cv2.waitKey(0)

cv2.destroyAllWindows()

'''
im = np.array(imageio.imread("cross_image.jpg"))

# Downsample the image to the (approximate) MBot resolution
im = im[::5,::5]

# im = im[100:300,100:300]
# im = im[600:1000,800:1500]
# im = im[:,:2400]
# im = im[700:900, 1600:1800]

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
plt.imsave("output.jpg", im)
'''
