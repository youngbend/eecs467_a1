import numpy as np
import math

def gradient(A, B):
    # Calculate an approximate gradient between two pixels values
    return int(A) - int(B)


class Target:
    def __init__(self, row, column, radius):
        # Initialize tracker instance at center location (x,y)
        self.loss_count = 0
        self.center_row = row
        self.center_col = column
        self.radius = radius
        self.prev_location = None


    def update(self, row, column, radius):
        self.prev_location = (self.center_row, self.center_col, self.radius)
        self.loss_count = 0
        self.center_row = row
        self.center_col = column
        self.radius = radius


    def lost(self):
        # Increment loss counter
        self.loss_count += 1


    def get_center(self):
        return (self.center_row, self.center_col)


class Tracker:
    def __init__(self, scan_offset, target_offset, threshhold, tracking_offset, tracking_timeout):
        # Initialize tracker with scanning offset (row_offset, col_offset),
        # target bound offset, and gradient threshhold

        self.__targets = []

        # (row_offset, col_offset) responsible for determining the distance
        # between samples within the global search.
        self.__scan_offset = scan_offset

        # offset around localized feature search that determines how far outside
        # the bounds of the previous layer we should search
        self.__target_offset = target_offset

        # Threshhold for what we consider to be a significant gradient
        self.__threshhold = threshhold

        # Base offset for determining spread of updated tracker search
        self.__tracking_offset = tracking_offset

        # Number of frames to look before removing tracked target
        self.__tracking_timeout = tracking_timeout


    def get_target_centers(self):
        # Return the center of all tracked targets
        return [t.get_center() for t in self.__targets]


    def scan(self, image, border=(10,10,10,10)):
        # Scan image for target. Scans horizontally from top to bottom
        # Border is the amount of pixels around the edges that is not scanned
        # border = (top_margin, bottom_margin, left_margin, right_margin)

        self.update_targets(image)

        for r in range(border[0], image.shape[0] - border[1], self.__scan_offset[0]):
            for c in range(border[2], image.shape[1] - border[3], self.__scan_offset[1]):

                # If we find a significant rising gradient (left side of the cross),
                # start a localized search to distinguish features from false positives
                if gradient(image[r,c-self.__scan_offset[1]], image[r,c]) > self.__threshhold:
                    inside_target = False
                    for t in self.__targets:
                        # Skip the region that has already had a target
                        if ((t.center_row - t.radius) < r < (t.center_row + t.radius)) and ((t.center_col - t.radius) < c < (t.center_col + t.radius)):
                            inside_target = True
                            break

                    if inside_target:
                        continue

                    # If the localized search finds a target, stop the global search
                    is_target, center_row, center_column, radius = self.pinpoint_target(image, r, c)
                    if is_target:
                        self.__targets.append(Target(center_row, center_column, radius))


    def update_targets(self, image):
        # Update all targets without scanning the whole image
        i = 0
        while i < len(self.__targets):
            row_offset = 0
            col_offset = 0
            depth_offset = 0
            tracking_offset = self.__tracking_offset
            if self.__targets[i].prev_location:
                row_offset = self.__targets[i].center_row - self.__targets[i].prev_location[0]
                col_offset = self.__targets[i].center_col - self.__targets[i].prev_location[1]
                depth_offset = self.__targets[i].radius - self.__targets[i].prev_location[2]
                depth_offset *= depth_offset > 0
                if self.__targets[i].loss_count:
                    interpolation_factor = 3 - 4 * np.exp(-self.__targets[i].loss_count / 2)
                    row_offset *= interpolation_factor
                    col_offset *= interpolation_factor
                    depth_offset *= interpolation_factor
                    tracking_offset *= interpolation_factor

            elif self.__targets[i].loss_count:
                tracking_offset *= 2

            top = int(self.__targets[i].center_row - self.__targets[i].radius + row_offset - depth_offset - tracking_offset)
            bottom = int(self.__targets[i].center_row + self.__targets[i].radius + row_offset + depth_offset + tracking_offset)
            left = int(self.__targets[i].center_col - self.__targets[i].radius + col_offset - depth_offset - tracking_offset)
            right = int(self.__targets[i].center_col + self.__targets[i].radius + col_offset + depth_offset + tracking_offset)

            if top < 0 or left < 0 or bottom > image.shape[0] or right > image.shape[1]:
                self.__targets.pop(i)
                continue

            # (DEBUG)
            # try:
            #     self.__rgb[top,left:right] = [0,255,0]
            #     self.__rgb[bottom,left:right] = [0,255,0]
            #     self.__rgb[top:bottom,left] = [0,255,0]
            #     self.__rgb[top:bottom,right] = [0,255,0]
            # except:
            #     None

            complete = False
            is_found = False
            center_row = 0
            center_col = 0
            radius = 0

            for r in range(top, bottom, self.__scan_offset[0]):
                for c in range(left + self.__scan_offset[1], right, self.__scan_offset[1]):

                    # If we find a significant rising gradient (left side of the cross),
                    # start a localized search to distinguish features from false positives
                    if gradient(image[r,c-self.__scan_offset[1]], image[r,c]) > self.__threshhold:

                        # If the localized search finds a target, stop the global search
                        is_found, center_row, center_column, radius = self.pinpoint_target(image, r, c)
                        if is_found:
                            complete = True
                            break

                if complete:
                    break

            if is_found:
                self.__targets[i].update(center_row, center_column, radius)

            else:
                self.__targets[i].lost()
                if self.__targets[i].loss_count > self.__tracking_timeout:
                    self.__targets.pop(i)
                    continue

            i += 1


    def pinpoint_target(self, image, row, col):
        # Pinpoint the target center given the triggering index

        # Define a smaller scan offset for the localized search so we can detect
        # finer features
        scan_offset = [math.ceil(self.__scan_offset[0] / 2), math.ceil(self.__scan_offset[1] / 2)]

        # Determine the initial scan width
        initial_left = None
        for c in range(col, 0, -scan_offset[1]):
            if gradient(image[row, c-scan_offset[1]], image[row,c]) > self.__threshhold:
                initial_left = c
                break

        if not initial_left:
            return (False, 0, 0, 0)

        initial_right = None
        for c in range(col, image.shape[1], scan_offset[1]):
            if gradient(image[row,c-scan_offset[1]], image[row,c]) < -self.__threshhold:
                initial_right = c
                break

        if not initial_right:
            return (False, 0, 0, 0)

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
        left_right = [initial_left, initial_right]
        gradient_counter = 0

        sum_bar_widths = initial_right - initial_left
        num_row_steps = 1

        # Iterate through the rows until the bottom of the image.
        for r in range(row - self.__target_offset, image.shape[0], scan_offset[0]):

            # If our search goes outside of the boundaries, return.
            if r < 0 or r >= image.shape[0]:
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
                if c < scan_offset[1] or c >= image.shape[1]:
                    return (False, 0, 0, 0)

                # Encounter left edge on first rising gradient
                if gradient(image[r,c-scan_offset[1]], image[r,c]) > self.__threshhold:
                    cross_encounter = True
                    if not edge_trigger:
                        edge_trigger = True
                        left_right[0] = c

                        # (DEBUG) Color rising edge blue
                        # try:
                        #     self.__rgb[r,c] = [0,0,255]
                        # except:
                        #     None

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
                        # try:
                        #     self.__rgb[r,c] = [0,255,0]
                        # except:
                        #     None

                        # Mark the top right corner of the bar
                        if top and not vbar_bounds[1]:
                            vbar_bounds[1] = c

                        continue

                # For checking if we encountered the cross within the row
                if top and image[r,c] < min_row_intensity:
                    min_row_intensity = image[r,c]
                    if min_row_intensity < min_intensity:
                        min_intensity = min_row_intensity

            if cross_encounter:
                gradient_counter = 0
                row_width = left_right[1] - left_right[0]
                if abs((sum_bar_widths / num_row_steps) - row_width) > (self.__target_offset / 2):
                    return (False, 0, 0, 0)
                num_row_steps += 1
                sum_bar_widths += row_width

            # If a black pixel was not encountered in this row, mark the bottom
            elif top and not ((min_row_intensity - min_intensity) < self.__threshhold):
                bottom = r - scan_offset[0]
                vbar_bounds[2] = left_right[0]
                vbar_bounds[3] = left_right[1]
                break

            else:
                gradient_counter += 1

            if gradient_counter > 1.5 * (initial_right - initial_left):
                return (False, 0, 0, 0)

        # If all of the features of the vertical bar were not detected, the feature
        # is probably not a cross
        if not top or not bottom or not all(vbar_bounds):
            return (False, 0, 0, 0)

        # If the top and bottom were not similar in width
        if abs((vbar_bounds[1] - vbar_bounds[0]) - (vbar_bounds[3] - vbar_bounds[2])) > self.__target_offset:
            return (False, 0, 0, 0)

        center_row = (top + bottom) // 2
        center_column = int(np.mean(vbar_bounds))
        column_radius = (vbar_bounds[1] - vbar_bounds[0]) // 2

        if (vbar_bounds[1] - vbar_bounds[0]) > 0.5 * (bottom - top):
            return (False, 0, 0, 0)

        left = None
        right = None

        # [top_left, bottom_left, top_right, bottom_right]
        hbar_bounds = [None, None, None, None]

        sum_bar_widths /= num_row_steps
        num_col_steps = 1

        # Perform a vertical scan from the center until we find the left edge
        up_down = [center_row - column_radius, center_row + column_radius]
        for c in range(center_column - column_radius, 0, -scan_offset[1]):

            if c < 0:
                return (False, 0, 0, 0)

            edge_trigger = False
            cross_encounter = False
            min_col_intensity = 255

            for r in range(up_down[0] - self.__target_offset, up_down[1] + self.__target_offset, scan_offset[0]):

                if r < 0 or r >= image.shape[0]:
                    return (False, 0, 0, 0)

                if gradient(image[r-scan_offset[0],c], image[r,c]) > self.__threshhold:
                    cross_encounter = True
                    if not edge_trigger:
                        edge_trigger = True
                        up_down[0] = r

                        # (DEBUG) Color rising edge blue
                        # try:
                        #     self.__rgb[r,c] = [0,0,255]
                        # except:
                        #     None

                elif gradient(image[r-scan_offset[0],c], image[r,c]) < -self.__threshhold:
                    cross_encounter = True
                    if edge_trigger:
                        up_down[1] = r

                        # (DEBUG) Color falling edge green
                        # try:
                        #     self.__rgb[r,c] = [0,255,0]
                        # except:
                        #     None

                        continue

                if image[r,c] < min_col_intensity:
                    min_col_intensity = image[r,c]

            if not cross_encounter and abs(min_col_intensity - min_intensity) > self.__threshhold:
                left = c
                hbar_bounds[0] = up_down[0]
                hbar_bounds[1] = up_down[1]
                break

        # Perform a vertical scan from the center until we find the right edge
        up_down = [center_row - column_radius, center_row + column_radius]
        for c in range(center_column + column_radius, image.shape[1], scan_offset[1]):

            if c >= image.shape[1]:
                return (False, 0, 0, 0)

            edge_trigger = False
            cross_encounter = False
            min_col_intensity = 255

            for r in range(up_down[0] - self.__target_offset, up_down[1] + self.__target_offset, scan_offset[0]):

                if r < 0 or r >= image.shape[0]:
                    return (False, 0, 0, 0)

                if gradient(image[r-scan_offset[0],c], image[r,c]) > self.__threshhold:
                    cross_encounter = True
                    if not edge_trigger:
                        edge_trigger = True
                        up_down[0] = r

                        # (DEBUG) Color rising edge blue
                        # try:
                        #     self.__rgb[r,c] = [0,0,255]
                        # except:
                        #     None

                elif gradient(image[r-scan_offset[0],c], image[r,c]) < -self.__threshhold:
                    cross_encounter = True
                    if edge_trigger:
                        up_down[1] = r

                        # (DEBUG) Color falling edge green
                        # try:
                        #     self.__rgb[r,c] = [0,255,0]
                        # except:
                        #     None

                        continue

                if image[r,c] < min_col_intensity:
                    min_col_intensity = image[r,c]

            if not cross_encounter and abs(min_col_intensity - min_intensity) > self.__threshhold:
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

        if (hbar_bounds[1] - hbar_bounds[0]) > 0.5 * (right - left):
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

        # Check that vectors are about the same length
        if abs(np.linalg.norm(up_vec) - np.linalg.norm(right_vec)) > self.__target_offset:
            return (False, 0, 0, 0)

        # Check that vectors are not tiny
        if np.linalg.norm(down_vec) < 2 * self.__target_offset or np.linalg.norm(left_vec) < 2 * self.__target_offset:
            return (False, 0, 0, 0)

        # (DEBUG)
        # print(up_vec)
        # print(down_vec)
        # print(right_vec)
        # print(left_vec)

        # Classify as not a cross if the angle is off by more than 15 degrees
        if abs(np.dot(up_unit, right_unit)) > 0.25:
            return (False, 0, 0, 0)

        if abs(np.dot(up_unit, left_unit)) > 0.25:
            return (False, 0, 0, 0)

        if abs(np.dot(left_unit, down_unit)) > 0.25:
            return (False, 0, 0, 0)

        if abs(np.dot(right_unit, down_unit)) > 0.25:
            return (False, 0, 0, 0)

        # Classify as not a cross if the opposite ends are not approximately parallel
        if abs(abs(np.dot(up_unit, down_unit)) - 1) > 0.1:
            return (False, 0, 0, 0)

        if abs(abs(np.dot(right_unit, left_unit)) - 1) > 0.1:
            return (False, 0, 0, 0)

        # (DEBUG) Draw lines on the bounds
        # try:
        #     self.__rgb[top,vbar_bounds[0]:vbar_bounds[1]] = [255,0,0]
        #     self.__rgb[bottom,vbar_bounds[2]:vbar_bounds[3]] = [255,0,0]
        #     self.__rgb[hbar_bounds[0]:hbar_bounds[1], left] = [255,0,0]
        #     self.__rgb[hbar_bounds[2]:hbar_bounds[3], right] = [255,0,0]
        # except:
        #     None

        return (True, center_row, center_column, max(bottom-top, right-left) // 2)

    def attach_rgb(self, rgb):
        # Link the RGB to the class object so we can draw on it.
        self.__rgb = rgb
