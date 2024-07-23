# imports
import numpy as np
import cv2 as cv
import glob

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 1, 0.001)

# Real world coordinates of circular grid (assuming grid is 8x18 and spacing is 20 units)
grid_width = 20  # spacing between circles
grid_height = 20
grid_rows = 8
grid_cols = 18

obj3d = np.zeros((grid_rows * grid_cols, 3), np.float32)
obj3d[:, :2] = np.mgrid[0:grid_rows, 0:grid_cols].T.reshape(-1, 2) * grid_width

print(obj3d)

# Vector to store 3D points
obj_points = []
# Vector to store 2D points
img_points = []

# Extracting path of individual image stored in a given directory
images = glob.glob('./img04.jpeg')
for f in images:
    # Loading image
    print(f)
    img = cv.imread(f)
    # Conversion to grayscale image
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # To find the position of circles in the grid pattern
    ret, corners = cv.findCirclesGrid(
        gray, (grid_rows, grid_cols), None, flags=cv.CALIB_CB_SYMMETRIC_GRID)
    print(ret)
    print(corners)

    # If true is returned,
    # then 3D and 2D vector points are updated and corner is drawn on image
    if True:
        obj_points.append(obj3d)

        corners2 = cv.cornerSubPix(gray, corners, (grid_rows, grid_cols), (-1, -1), criteria)
        # In case of circular grids,
        # the cornerSubPix() is not always needed, so alternative method is:
        # corners2 = corners
        img_points.append(corners2)

        # Drawing the corners, saving and displaying the image
        cv.drawChessboardCorners(img, (grid_rows, grid_cols), corners2, ret)
        cv.imwrite('output.jpg', img)  #To save corner-drawn image
        cv.imshow('img', img)
        cv.waitKey(0)
cv.destroyAllWindows()


if len(obj_points) > 0 and len(img_points) > 0:
    print("**********************************")
"""Camera calibration: 
Passing the value of known 3D points (obj points) and the corresponding pixel coordinates 
of the detected corners (img points)"""
ret, camera_mat, distortion, rotation_vecs, translation_vecs = cv.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

print("Error in projection : \n", ret)
print("\nCamera matrix : \n", camera_mat)
print("\nDistortion coefficients : \n", distortion)
print("\nRotation vector : \n", rotation_vecs)
print("\nTranslation vector : \n", translation_vecs)

# Undistort and show the image
for f in images:
    img = cv.imread(f)
    h, w = img.shape[:2]
    new_camera_mtx, roi = cv.getOptimalNewCameraMatrix(camera_mat, distortion, (w, h), 1, (w, h))

    # Undistort
    undistorted_img = cv.undistort(img, camera_mat, distortion, None, new_camera_mtx)

    # Crop the image
    x, y, w, h = roi
    undistorted_img = undistorted_img[y:y+h, x:x+w]

    # Save and show the undistorted image
    cv.imwrite('undistorted_output.jpg', undistorted_img)
    cv.imshow('undistorted_img', undistorted_img)
    cv.waitKey(0)
cv.destroyAllWindows()