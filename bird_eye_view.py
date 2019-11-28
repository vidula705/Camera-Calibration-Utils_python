import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Select the points on the image
# to go back and forth between
# driver and bird eye perspective
def get_bird_eye_view_transform(img_size):

    src = np.float32(
        [[(img_size[0] / 2) - 62, img_size[1] / 2 + 100],
        [((img_size[0] / 6) - 10), img_size[1]],
        [(img_size[0] * 5 / 6) + 60, img_size[1]],
        [(img_size[0] / 2 + 68), img_size[1] / 2 + 100]])
    dst = np.float32(
        [[(img_size[0] / 4), 0],
        [(img_size[0] / 4), img_size[1]],
        [(img_size[0] * 3 / 4), img_size[1]],
        [(img_size[0] * 3 / 4), 0]])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    return M, Minv, src, dst

# Warp the image based
# on selected matrix M
# Used for both warp and unwarp
def warp(img, M):
    img_size = (img.shape[1], img.shape[0])
    warped_img = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped_img

#
# Unit test
#
if __name__ == '__main__':

    # Read camera calibration parameters
    from camera_calibration import get_calibration_parameters
    filename = "camera_parameters_pickle.p"
    mtx, dist = get_calibration_parameters(filename)

    img_size = (1280,720)
    M, Minv, src, dst = get_bird_eye_view_transform(img_size)

    filename = '..\\test_images\\straight_lines2.jpg'
    img = mpimg.imread(filename)

    # Undistort using mtx and dist
    undist_img = cv2.undistort(img, mtx, dist, None, mtx)

    print(src[0][0], src[0][1])
    cv2.line(undist_img, (src[0][0], src[0][1]), (src[1][0], src[1][1]), (255,0,0), 2)
    cv2.line(undist_img, (src[1][0], src[1][1]), (src[2][0], src[2][1]), (255,0,0), 2)
    cv2.line(undist_img, (src[2][0], src[2][1]), (src[3][0], src[3][1]), (255,0,0), 2)
    cv2.line(undist_img, (src[3][0], src[3][1]), (src[0][0], src[0][1]), (255,0,0), 2)

    # Get the bird eye view image
    warped = warp(undist_img, M)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()

    ax1.imshow(undist_img)
    ax1.set_title('Undistorted', fontsize=20)

    ax2.imshow(warped)
    ax2.set_title('Bird Eye View', fontsize=20)

    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()




