# Driver Unwanted Pose Detection

A computer-vision prototype that monitors a driver using a webcam or
recorded video. It flags selected behaviours: sustained sideways
leaning, prolonged eye closure, and mouth opening.

> **Disclaimer:** This is an educational prototype, not a certified
> safety system. Do not rely on it to determine whether someone is safe
> to drive.

## Features

-   Webcam and recorded-video input.
-   Body posture monitoring using pose landmarks.
-   Eye-closure estimation using Eye Aspect Ratio (EAR).
-   Mouth-opening estimation using Mouth Aspect Ratio (MAR).
-   Visual alerts, Windows alert sounds, and CSV event logging.
-   Short calibration period to establish a posture baseline.

## Tech Stack

  Technology          Use
  ------------------- ----------------------------------------------
  Python              Main programming language
  OpenCV              Video capture, frame processing, and display
  MediaPipe Tasks     Pose and face landmark detection
  NumPy               Numerical and landmark calculations
  `argparse`          Command-line input selection
  `csv`, `datetime`   Timestamped event logging
  `winsound`          Alert sounds on Windows

## Requirements

-   Windows 10/11
-   Python version supported by your installed MediaPipe release
-   VS Code and the official Microsoft Python extension (recommended)
-   Webcam or recorded test video
-   Internet access for initial installation and model downloads

**Python compatibility:** If MediaPipe will not install on Python 3.15,
use a Python version supported by the current MediaPipe release (for
example, Python 3.11 or 3.12) and select that interpreter in VS Code.

## What to Download

1.  **Python:**
    [python.org/downloads](https://www.python.org/downloads/)
2.  **VS Code:** [code.visualstudio.com](https://code.visualstudio.com/)
3.  **VS Code Python extension:** Install the Microsoft Python extension
    from the Extensions panel.
4.  **Python packages:** Install them using the commands below.
5.  **MediaPipe model files:** Download the `.task` files required by
    your code:
    -   `pose_landmarker_full.task` --- [Pose Landmarker
        documentation](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker)
    -   `face_landmarker.task` --- [Face Landmarker
        documentation](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)

Place both model files where your script expects them, and make sure the
paths in `driver_monitor.py` match.

## Installation

Open the project folder in VS Code. If using Git, clone your repository
first:

``` bash
git clone <YOUR_REPOSITORY_URL>
cd driver-unwanted-pose
```

Replace the example URL and folder name with your actual repository
details. If the project is already on your computer, simply open its
folder.

Create and activate a virtual environment in PowerShell:

``` powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

``` bash
python -m pip install --upgrade pip
python -m pip install opencv-python mediapipe numpy
```

`argparse`, `csv`, and `datetime` are included with Python. `winsound`
is included with Windows Python and does not need a separate install.

## Run the Project

**Start webcam monitoring:**

``` bash
python driver_monitor.py
```

**Monitor a recorded video:**

``` bash
python driver_monitor.py --source "test_video.mp4"
```

Replace `test_video.mp4` with your video's filename or full path. Follow
the argument names in your script if they differ. Press **Q** in the
video window to quit, if supported by the script.

## How the System Works

The current system follows this processing pipeline:

```text
Webcam
   |
   v
Capture Video Frame
   |
   v
Convert BGR to RGB
   |
   v
MediaPipe Pose Landmarker
   |
   v
Extract Body Landmarks
   |
   v
Identify Shoulders and Hips
   |
   v
Calculate Torso Tilt
   |
   v
Apply Experimental Threshold
   |
   v
Display Status and Warning
```

### 1. Video capture

OpenCV captures frames from the webcam using `cv2.VideoCapture(0)`.

### 2. Image conversion

OpenCV captures frames in BGR format. The frames are converted to RGB before being passed to MediaPipe.

### 3. Landmark extraction

MediaPipe Pose Landmarker estimates 33 body landmarks, including the shoulders, hips, elbows, wrists, knees, and ankles.

### 4. Torso alignment

The midpoint between the two shoulders and the midpoint between the two hips are calculated. These midpoints are used to estimate the torso's orientation in the image.

### 5. Rule-based warning

The calculated angle is compared with an initial threshold. If the absolute angle exceeds the threshold for a specified duration, the system displays a warning.


## Current Detection Settings

The following are the current prototype settings in `driver_monitor.py`.
Adjust them carefully during testing.

  Setting                 Value
  ----------------------- ----------------
  `LEAN_THRESHOLD`        `20.0` degrees
  `LEAN_DURATION`         `1.0` second
  `EAR_THRESHOLD`         `0.20`
  `EYE_CLOSED_DURATION`   `1.5` seconds
  `MAR_THRESHOLD`         `0.35`
  `MOUTH_OPEN_DURATION`   `1.0` second
  Posture calibration     3 seconds

These values are experimental and may vary with lighting, camera angle,
distance, and individual differences. Mouth opening alone does not
confirm yawning.

## Event Log

The CSV log can include:

-   Timestamp and event type (`POSTURE_LEAN`, `EYE_CLOSED`, or
    `MOUTH_OPEN`)
-   EAR and MAR measurements
-   Estimated posture angle
-   Data-status information, if provided by the script

Use the log to review events and tune thresholds.

## Troubleshooting

-   **Camera unavailable:** Close other apps using it and check the
    camera index.
-   **Video not found:** Verify the file path, spelling, and extension.
-   **MediaPipe installation error:** Check the active interpreter and
    use a supported Python version.
-   **Model error:** Verify the `.task` files and configured paths.
-   **No sound:** Check Windows volume/output device and test `winsound`
    independently.
-   **No alerts:** Make sure the behaviour lasts longer than the
    configured duration; review thresholds.
-   **Unreliable landmarks:** Improve lighting and keep the face and
    upper body visible.

## Limitations and Safety

-   This prototype uses landmark geometry and thresholds; it does not
    detect every possible unwanted pose.
-   Results can be affected by occlusion, lighting, camera placement,
    and landmark errors.
-   Use consent when testing with people and do not share identifiable
    recordings without permission.
-   Test only while stationary or with recorded videos---never operate
    the interface while driving.
-   Alerts are experimental indicators, not proof of impairment or a
    substitute for attentive driving.


/////   for running video //////

python driver_monitor.py --source "C:\Python\driver-unwanted-pose\test_video.mp4"