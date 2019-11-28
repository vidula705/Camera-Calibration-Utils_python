import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle

#
# Read the pickle file to retrieve intrinsic and
# lens distortion parameters
#
def get_calibration_parameters(filename):
    # Read in the saved camera matrix and distortion coefficients
    dist_pickle = pickle.load( open(filename, "rb"))
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]
    return mtx, dist

#
# Generate 3D points for specified checker board pattern
#
def generate_3D_grid_points(num_inner_x, num_inner_y):

    # Prepare object points
    objp = np.zeros((num_inner_x*num_inner_y, 3), np.float32)
    objp[:, :2] = np.mgrid[0:num_inner_x,0:num_inner_y].T.reshape(-1, 2) # x, y corordinates

    return objp

#
# Find 2D points for specified image of checker board pattern
#
def get_2D_image_points(img, num_inner_x, num_inner_y):

    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (num_inner_x,num_inner_y), None)

    return ret, corners


#
# Read all images from the specified folder
#
def read_calibration_images(folder_path):

    images = [] # Storage for images used for calibration

    filenames = glob.glob(folder_path+'*.jpg')
    for filename in filenames:
        img = mpimg.imread(filename)
        images.append(img)

    return images

#
# Unit test
#
if __name__ == '__main__':

    objpoints = [] # 3D Real world points
    imgpoints = [] # 2D Points in image plane

    folder_path = "..\\camera_cal\\"
    images = read_calibration_images(folder_path)

    num_inner_x = 9
    num_inner_y = 6
    objp = generate_3D_grid_points(num_inner_x, num_inner_y)

    # Find corners in images and collect data for calibration
    img_shape = None
    for img in images:
        img_shape = img.shape
        ret, corners = get_2D_image_points(img, num_inner_x, num_inner_y)
        if ret is True:
            imgpoints.append(corners)
            objpoints.append(objp)
            # img = cv2.drawChessboardCorners(img, (num_inner_x,num_inner_y), corners, ret)
            # plt.imshow(img)
            # plt.show()

    # Calibrate the camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape[0:2], None, None)

    # Store the camera matrix and distortion coefficients
    file_handler = open("camera_parameters_pickle.p", "wb" )
    camera_params = {"mtx": mtx, "dist": dist}
    pickle.dump(camera_params, file_handler)

    # file_handler.seek(0)
    # camera_params = pickle.load( open("camera_parameters_pickle.p", "rb") )
    # mtx1 = camera_params["mtx"]
    # dist1 = camera_params["dist"]
    # print(mtx)
    # print(mtx1)
    # print(dist)
    # print(dist1)
