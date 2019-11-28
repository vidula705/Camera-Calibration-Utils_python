import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Define a function that applies Sobel x and y,
# then computes the magnitude of the gradient
# and applies a threshold
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):

    # 1) Convert to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

    # 3) Calculate the magnitude
    mag_sobel = np.sqrt(sobelx*sobelx + sobely*sobely)

    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255*mag_sobel/np.max(mag_sobel))

    # 5) Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1

    # 6) Return this mask as your binary_output image
    return binary_output

# Define a function that applies Sobel x and y,
# then computes the direction of the gradient
# and applies a threshold.
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):

    # 1) Convert to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)

    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    orient_sobel = np.arctan2(abs_sobely, abs_sobelx)

    # 5) Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(orient_sobel)
    binary_output[(orient_sobel >= thresh[0]) & (orient_sobel <= thresh[1])] = 1

    # 6) Return this mask as your binary_output image
    return binary_output

#
# Define a function that thresholds the S-channel of HLS
# Use exclusive lower bound (>) and inclusive upper (<=)
def hls_select(img, thresh=(0, 255)):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_img = hls[:,:,2]

    # 2) Apply a threshold to the S channel
    binary_output = np.zeros_like(s_img)
    binary_output[(s_img > thresh[0]) & (s_img <= thresh[1])] = 1

    # 3) Return a binary image of threshold result
    return binary_output

#
# Picked up from online blog
#
def color_threshold(img):

    # 1) Convert to HSV color space
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # 2) Generate mask for yellow lane
    yellow_min = np.array([15, 100, 120], np.uint8)
    yellow_max = np.array([80, 255, 255], np.uint8)
    yellow_mask = cv2.inRange(img, yellow_min, yellow_max)

    # 2) Generate mask for white lane
    white_min = np.array([0, 0, 200], np.uint8)
    white_max = np.array([255, 30, 255], np.uint8)
    white_mask = cv2.inRange(img, white_min, white_max)

    # 3) Merge Yellow and White Lane masks using OR operation
    binary_output = np.zeros_like(img[:, :, 0])
    binary_output[((yellow_mask != 0) | (white_mask != 0))] = 1

    filtered = img
    filtered[((yellow_mask == 0) & (white_mask == 0))] = 0

    return binary_output

#
# Use grad mag, orientation and image color
# to get the threshold
def combined_threshold(image):

    mag_binary = mag_thresh(image, sobel_kernel=15, mag_thresh=(50, 255))
    dir_binary = dir_threshold(image, sobel_kernel=15, thresh=(0.7, 1.2))
    # color_binary = hls_select(image, thresh=(90, 255))
    color_binary = color_threshold(image)

    combined = np.zeros_like(mag_binary)
    combined[((color_binary == 1) & ((mag_binary == 1) | (dir_binary == 1)))] = 1

    return combined, mag_binary, dir_binary, color_binary

#
# Unit test
#
if __name__ == '__main__':

    # Read test image
    image = mpimg.imread('..\\test_images\\test6.jpg')

    combined, mag_binary, dir_binary, hls_binary = combined_threshold(image)
    # color_binary = np.dstack(( mag_binary, dir_binary, hls_binary))

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(24, 9))
    f.tight_layout()

    ax1.imshow(combined, cmap='gray')
    ax1.set_title('Combined Mask', fontsize=20)

    ax2.imshow(mag_binary, cmap='gray')
    ax2.set_title('Gradient Magnitude Mask', fontsize=20)

    ax3.imshow(dir_binary, cmap='gray')
    ax3.set_title('Gradient Orientation Mask', fontsize=20)

    ax4.imshow(hls_binary, cmap='gray')
    ax4.set_title('Color Mask', fontsize=20)

    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()
