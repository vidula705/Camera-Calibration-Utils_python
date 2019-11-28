import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import binarize_image
import camera_calibration
import bird_eye_view
import localize_lanes
from moviepy.video.io.VideoFileClip import VideoFileClip

# Read camera calibration parameters
filename = "camera_parameters_pickle.p"
mtx, dist = camera_calibration.get_calibration_parameters(filename)

# Get the perspective transformation matrices
img_size = (1280,720)
M, Minv, src, dst = bird_eye_view.get_bird_eye_view_transform(img_size)

ctr = [0]
old_curvature_m = [0]
old_shift_m = [0]

#
#
#
def test_using_video():

    video_output = '..\\project_video_out.mp4'
    clip1 = VideoFileClip('..\\project_video.mp4')

    ctr[0] = 0
    video_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    video_clip.write_videofile(video_output, audio=False)


#
# Called for every frame
#
def process_image(img):

    # Undistort using mtx and dist
    undist_img = cv2.undistort(img, mtx, dist, None, mtx)

    # Threshold image using gradient and color
    combined, mag_binary, dir_binary, hls_binary = binarize_image.combined_threshold(img)

    # Get the bird eye view image
    binary_warped = bird_eye_view.warp(combined, M)

    # Fit polynomial to piecewise filtered lane lines
    left_fit, right_fit = localize_lanes.piecewise_fit_lane(binary_warped)

    # Fit polynomial in case previous frame polynomial fit is already known
    left_fit, right_fit = localize_lanes.incremental_lane_search(binary_warped, left_fit, right_fit)

    # Calculate the average curvature from both lanes
    curvature_m, car_shift_m = localize_lanes.calculate_curvature(binary_warped, left_fit, right_fit)

    # Draw the overlay of detected lanes
    result_img = localize_lanes.draw_lane_driver_eye_view(undist_img, left_fit, right_fit, Minv)

    # Make the display readable by writing only on every 5th frame
    if ctr[0] % 5 is 0:
        old_curvature_m[0] = curvature_m
        old_shift_m[0] = car_shift_m
        cv2.putText(result_img, "Lane Curvature: {}m".format(curvature_m), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    color=(255, 255, 255), thickness=2)
        cv2.putText(result_img, "Car is {}m off from center".format(car_shift_m), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    color=(255, 255, 255), thickness=2)
    else:
        curvature_m = old_curvature_m[0]
        car_shift_m = old_shift_m[0]
        cv2.putText(result_img, "Lane Curvature: {}m".format(curvature_m), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    color=(255, 255, 255), thickness=2)
        cv2.putText(result_img, "Car is {}m off from center".format(car_shift_m), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    color=(255, 255, 255), thickness=2)


    ctr[0] += 1

    return result_img


#
#
#
if __name__ == '__main__':

    test_using_video()


