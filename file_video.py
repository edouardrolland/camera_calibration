#!/usr/bin/env python

"""file_video.py: Perform camera calibration using a video file containing a chessboard pattern."""
__author__      = "Edouard G. A. Rolland, Kilian Meier"

# More information here: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
# https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html

import numpy as np
import cv2 as cv

# --- Global parameters ---
IS_CAP_STREAM = False  # Reading from a video file
SHOW_FRAMES = True

# Chessboard size and square size in meters
GRID_SHAPE = (9, 14)
SQUARE_SIZE = 16.5e-3

# Video file used for calibration
CALLIB_VIDEO = "ressources/test_camille.mp4"

# --- Preparing the 3D points of the chessboard ---
# Generate a 3D points array for each corner of the grid
objp = np.zeros((GRID_SHAPE[0] * GRID_SHAPE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0 : GRID_SHAPE[0], 0 : GRID_SHAPE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

# Storing points in two lists:
#  - objpoints: 3D real-world points
#  - imgpoints: 2D image points
objpoints = []
imgpoints = []

# --- Window configuration for display ---
if SHOW_FRAMES:
    cv.namedWindow('img', cv.WND_PROP_FULLSCREEN)
    cv.resizeWindow("img", 1280, 720)

# --- Opening the video file ---
cap = cv.VideoCapture(CALLIB_VIDEO)

while True:
    # Read one frame from the video
    ret, frame = cap.read()
    if not ret:
        # End of the video or reading error
        break

    # Convert to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detect the corners of the chessboard
    found_corners, corners = cv.findChessboardCornersSB(gray, GRID_SHAPE)

    if found_corners:
        # Automatically store the frame if corners are found
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw the detected corners for visualization
        cv.drawChessboardCorners(frame, GRID_SHAPE, corners, found_corners)
        print(f"Frame added! Total number of frames: {len(imgpoints)}")

    # Optional display
    if SHOW_FRAMES:
        cv.imshow('img', frame)

    # Keyboard interaction
    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        # Quit when 'q' is pressed
        break

cap.release()
cv.destroyAllWindows()

# --- Final calibration if at least one valid frame is available ---
if len(objpoints) > 0 and len(imgpoints) > 0:
    # Compute camera parameters
    ret_calib, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints,
        imgpoints,
        gray.shape[::-1],
        None,
        None,
        flags=cv.CALIB_FIX_ASPECT_RATIO + cv.CALIB_FIX_PRINCIPAL_POINT
    )

    # Display results
    print(f"RMS reprojection error: {ret_calib}")
    print("Camera matrix:")
    print(mtx)
    print("Distortion coefficients:")
    print(dist)
else:
    print("No valid frames were captured for calibration.")
