import numpy as np, matplotlib as mpl, matplotlib.pyplot as plt
import imageio

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
    def __init__(self, scan_offset, scan_border, target_offset, contrast_threshhold):
        # Initialize tracker with scanning offset (row_offset, col_offset),
        # scanning border (row_border, col_border), target bound offset,
        # and contrast_threshhold
        self.__targets = []
        self.__scan_offset = scan_offset
        self.__border = scan_border
        self.__target_offset = target_offset
        self.__contrast_threshhold = contrast_threshhold

    def get_target_centers(self):
        return [t.get_center() for t in self.__targets]

    def scan(self, image):
        # Scan image for target. Scans horizontally from top to bottom
        for r in range(self.__border[0], image.shape[0] - self.__border[0], self.__scan_offset[0]):
            for c in range(self.__border[1], image.shape[1] - self.__border[1], self.__scan_offset[1]):
                if Tracker.contrast_ratio(image[r,c-self.__scan_offset[1]], image[r,c]) > self.__contrast_threshhold:
                    if self.pinpoint_target(image, r, c):
                        return
                    # image[r,c-self.__scan_offset[1]] = [255,0,0]

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
                if Tracker.contrast_ratio(image[r,c-1], image[r,c]) > self.__contrast_threshhold:
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
        
        center = (top + bottom) / 2
        
        # 1. find horizontal bar - start from center -> move left within radius
        # 2. if no bar found within the radius, immediately return False

        # image[top,:] = [255,0,0]
        # image[bottom,:] = [255,0,0]
        return True

    def luminocity(p):
        pix = p / 255.0
        transform = np.vectorize(lambda x: (x / 12.92) if (x <= 0.03928) else (((x + 0.055) / 1.055)**2.4))
        pix = transform(pix)
        coefs = np.array([0.2126, 0.7152, 0.0722])
        return np.dot(coefs, pix)

    def contrast_ratio(A, B):
        # Calculate the contrast ratio between two np.array pixels
        lumA = Tracker.luminocity(A)
        lumB = Tracker.luminocity(B)
        return (lumA + 0.05) / (lumB + 0.05) if lumA > lumB else (lumB + 0.05) / (lumA + 0.05)


tracker = Tracker((2,2), (10,10), 5, 1.7)

im = np.array(imageio.imread("tracking_dark.jpg"))
im = im[::5,::5]
im = im[100:300,100:300]

plt.figure()
plt.title('Image')
plt.axis('off')
plt.imshow(im)

tracker.scan(im)

plt.figure()
plt.title('High contrast')
plt.axis('off')
plt.imshow(im)
plt.imsave("points_of_interest.jpg", im)
