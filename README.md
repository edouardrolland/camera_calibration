# Camera Calibration

This repository contains a script that performs **camera calibration** using a video file containing a chessboard pattern. The script automatically detects the chessboard corners in each video frame, collects both the 3D real-world coordinates and their corresponding 2D image coordinates, and then computes the camera calibration parameters (camera matrix and distortion coefficients).

---

## How It Works

1. **Chessboard Detection**  
   The script reads each frame from the provided video file and uses OpenCV’s `cv.findChessboardCornersSB` function to detect the chessboard corners.
   
2. **Automatic Frame Collection**  
   Whenever the chessboard is detected in a frame, the 3D coordinates (`objp`) and the detected 2D corners (`corners`) are automatically saved in two lists: 
   - `objpoints`: A list of arrays containing the 3D real-world points for each detected chessboard.  
   - `imgpoints`: A list of arrays containing the 2D image coordinates for each detected chessboard.

3. **Calibration Computation**  
   Once the entire video has been processed (or the user quits by pressing **q**), if at least one valid detection was recorded, OpenCV’s `cv.calibrateCamera` function is called to compute the camera matrix and distortion coefficients.

4. **Results**  
   The script prints:
   - **RMS Reprojection Error**: A measure of how well the detected points fit the computed camera model.  
   - **Camera Matrix**: Intrinsic parameters of the camera, including focal lengths and principal point.  
   - **Distortion Coefficients**: Radial and tangential distortion coefficients.

---

## Requirements

- **Python 3**  
- **OpenCV 4.x** (including the `cv2` Python module)  
- **NumPy**

Use the following command to install the required packages if needed:
```bash
pip install opencv-python numpy
```

---

## Script Overview

- **Global Parameters**  
  - `IS_CAP_STREAM (bool)`: Toggle between using a video file or a live camera feed (not used in the default script flow).  
  - `SHOW_FRAMES (bool)`: Whether to display frames in a window during processing.  
  - `GRID_SHAPE (tuple)`: The number of *inner* corners (rows, columns) on the chessboard.  
  - `SQUARE_SIZE (float)`: The physical size of each chessboard square, in meters.  
  - `CALLIB_VIDEO (str)`: Path to the video file for calibration.

- **Variables**  
  - `objp (ndarray)`: A single array of 3D points for the chessboard corners.  
  - `objpoints (list)`: Accumulated 3D points for all valid frames.  
  - `imgpoints (list)`: Accumulated 2D image points for all valid frames.  

- **Main Functions**  
  - `cv.findChessboardCornersSB`: Detects chessboard corners in a grayscale image.  
  - `cv.drawChessboardCorners`: Visualizes the detected corners.  
  - `cv.calibrateCamera`: Computes the final calibration (camera matrix & distortion coefficients).  

---

## Usage

1. **Place Your Video**  
   Edit `CALLIB_VIDEO` in the script or place your video file in the correct path so that `CALLIB_VIDEO` points to it:
   ```python
   CALLIB_VIDEO = "path/to/your_chessboard_video.mp4"
   ```

2. **Run the Script**  
   From your terminal or command prompt:
   ```bash
   python file_video.py
   ```
   - The script will open a display window (if `SHOW_FRAMES` is `True`) and start reading the video.  
   - Each time a valid chessboard is detected, a message is printed (`Frame added! ...`).

3. **Quit**  
   - Press **q** to stop the script at any time.

4. **Calibration Results**  
   - At the end of the video (or when you press **q**), the script will compute the calibration if it has at least one valid detection.  
   - It prints the **RMS Reprojection Error**, **Camera Matrix**, and **Distortion Coefficients**.

---

## Notes & Tips

1. **Chessboard Requirements**  
   - Ensure that the chessboard used has the same number of *inner* corners as specified by `GRID_SHAPE`.  
   - For `GRID_SHAPE = (9, 14)`, the actual chessboard has 10 x 15 squares, but 9 x 14 inner corners.

2. **Video Quality**  
   - Proper lighting and minimal motion blur will improve corner detection reliability.  
   - If the chessboard is partially out of the frame or the image is too blurry, OpenCV may fail to detect the corners.

3. **Adjustments**  
   - If you want to tune calibration flags (e.g., fix different camera parameters or refine extrinsics), modify the `flags` parameter in `cv.calibrateCamera`.
   - If you want to see frames during processing but in a smaller or bigger window, edit the lines:
     ```python
     cv.namedWindow('img', cv.WND_PROP_FULLSCREEN)
     cv.resizeWindow("img", 1280, 720)
     ```
   - You can also set `SHOW_FRAMES` to `False` to disable the display.

4. **Resources**  
   - [OpenCV Camera Calibration Documentation](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)

---

## Example Output

An example of final output might look like:
```
Frame added! Total number of frames: 1
...
Frame added! Total number of frames: 7
RMS reprojection error: 0.323123
Camera matrix:
[[1.06017536e+03 0.00000000e+00 9.53719306e+02]
 [0.00000000e+00 1.05893628e+03 5.40939057e+02]
 [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
Distortion coefficients:
[[-0.28340811  0.07395907  0.00019359  0.00001714  0.00000000]]
```
In this example, the **RMS reprojection error** is around 0.32, which is acceptable for many applications, and you can see the resulting **camera matrix** and **distortion coefficients**.

---

## License

This project is distributed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

**Authors**:  
- Edouard G. A. Rolland  
- Kilian Meier  

For more information, consult the [OpenCV camera calibration documentation](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)