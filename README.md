# Driver Unwanted Pose Detection System

A real-time computer vision project that uses Python, OpenCV, and MediaPipe to detect human body landmarks and identify potentially unwanted driver postures through webcam input.

The project is being developed as an experimental driver-monitoring system, with future extensions for facial behavior analysis, drowsiness detection, and recorded-video processing.

## Table of Contents

- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Current Features](#current-features)
- [Technology Stack](#technology-stack)
- [System Requirements](#system-requirements)
- [Project Structure](#project-structure)
- [Installation and Setup](#installation-and-setup)
- [Downloading the Pose Model](#downloading-the-pose-model)
- [Running the Project](#running-the-project)
- [How the System Works](#how-the-system-works)
- [Posture Detection Logic](#posture-detection-logic)
- [Troubleshooting](#troubleshooting)
- [Current Limitations](#current-limitations)
- [Future Enhancements](#future-enhancements)
- [Disclaimer](#disclaimer)

## Project Overview

Driver Unwanted Pose Detection is a computer vision project designed to analyze a person's body posture using a webcam. It uses MediaPipe Pose Landmarker to identify important body landmarks, including the shoulders and hips, and OpenCV to process video frames and display visual feedback.

The current implementation focuses on live webcam-based body landmark detection and experimental torso-tilt detection.

The long-term objective is to develop a driver-monitoring system that can identify potentially unsafe posture and facial behavior over time.

## Objectives

1. Detect body landmarks from live webcam video.
2. Estimate torso alignment using shoulder and hip positions.
3. Identify sustained sideways torso tilt using configurable rules.
4. Display posture status and warnings in real time.
5. Establish a foundation for facial behavior and drowsiness detection.
6. Extend the system to recorded videos and more robust posture classification.

## Current Features

- Live webcam capture using OpenCV.
- Human body landmark detection using MediaPipe.
- Visualization of selected shoulder and hip landmarks.
- Torso-angle calculation relative to the image's vertical direction.
- Configurable tilt threshold and warning duration.
- On-screen posture status and experimental warning messages.
- Python-based implementation suitable for further development.

**Current status:** Initial webcam landmark detection has been tested successfully. The posture-detection stage has been implemented for testing; its accuracy and thresholds still need to be evaluated and calibrated.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python 3.15.0 | Programming language |
| OpenCV 5.0.0 | Webcam access, image processing, and display |
| MediaPipe 1.1.1 | Body landmark detection |
| MediaPipe Pose Landmarker | Full-body pose estimation |
| VS Code | Development environment |
| PowerShell | Running commands on Windows |

These are the versions reported by the current development environment. Other compatible versions may work, but the project has not been tested across all environments.

## System Requirements

- Windows 10 or Windows 11, 64-bit.
- Python installation with `pip`.
- Visual Studio Code.
- A working webcam.
- Internet access for installing packages and downloading the model.
- A computer capable of running real-time image processing.

Python 3.15.0 is the version currently used in development. Package compatibility can change as Python and its dependencies evolve.

## Project Structure

The project currently uses the following structure:

```text
driver-unwanted-pose/
│
├── test_pose.py
├── pose_detection.py
├── posture_detection.py
├── pose_landmarker_full.task
├── README.md
└── .gitignore
```

### File descriptions

- `test_pose.py` — Checks the installed OpenCV and MediaPipe versions and inspects the available pose-detection API.
- `pose_detection.py` — Performs initial live webcam pose detection and displays body landmarks.
- `posture_detection.py` — Implements experimental torso-angle measurement and sustained-lean warnings.
- `pose_landmarker_full.task` — Pretrained MediaPipe model required for body landmark detection.
- `README.md` — Project documentation.
- `.gitignore` — Specifies files that Git should ignore.

## Installation and Setup

### Step 1: Install Python

Download Python from the official website:

https://www.python.org/downloads/

The current development environment uses Python 3.15.0.

During installation, enable the option to add Python to PATH if the installer provides it.

Verify the installation:

```powershell
python --version
```

Expected output in the current environment:

```text
Python 3.15.0
```

Check registered Python installations:

```powershell
py -0p
```

### Step 2: Install Visual Studio Code

Download VS Code:

https://code.visualstudio.com/

Install the **Python** extension published by Microsoft from the Extensions panel.

### Step 3: Open the project folder

Open VS Code and select:

**File → Open Folder**

Choose the project directory:

```text
C:\Python\driver-unwanted-pose
```

Open a terminal using:

**Terminal → New Terminal**

Navigate to the project directory if necessary:

```powershell
cd C:\Python\driver-unwanted-pose
```

### Step 4: Install the required libraries

Install OpenCV and MediaPipe using:

```powershell
python -m pip install --upgrade pip
python -m pip install mediapipe opencv-python
```

The current environment reports these packages as installed:

```text
mediapipe 1.1.1
opencv-python 5.0.0.93
```

MediaPipe also installs its required dependencies.

The original attempt to install `mediapipe==0.10.9` failed in the Python 3.15 environment. The current project instead uses the available MediaPipe 1.1.1 release.

### Step 5: Verify the installation

Run:

```powershell
python -c "import cv2, mediapipe as mp; print('OpenCV:', cv2.__version__); print('MediaPipe:', mp.__version__)"
```

Expected output:

```text
OpenCV: 5.0.0
MediaPipe: 1.1.1
```

If both version numbers appear, both packages can be imported successfully.

## Downloading the Pose Model

MediaPipe Pose Landmarker requires a model asset to perform pose estimation.

Download the official Full Pose Landmarker model:

https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task

Save the downloaded file in the root of the project directory.

The file must be named:

```text
pose_landmarker_full.task
```

The final location should be:

```text
C:\Python\driver-unwanted-pose\pose_landmarker_full.task
```

The Python scripts expect this model file to be available in the current working directory.

**Important:** The model is a pretrained asset. It is not a model trained specifically for driver posture classification.

## Running the Project

### 1. Test the MediaPipe installation

Run:

```powershell
python test_pose.py
```

The test script displays the OpenCV version, MediaPipe version, and available MediaPipe vision modules.

### 2. Run initial pose detection

Execute:

```powershell
python pose_detection.py
```

The program should open the webcam and display detected body landmarks.

Press `Q` to close the webcam window.

### 3. Run posture detection

Execute:

```powershell
python posture_detection.py
```

The program attempts to detect body landmarks and calculate torso tilt using shoulder and hip positions.

Depending on the detected posture, the application displays a status such as:

- `No pose detected`
- `Move into view`
- `Torso alignment: normal`
- `Leaning detected - checking`
- `WARNING: Sustained leaning`

Press `Q` to exit.

For meaningful testing, keep your upper body and hips visible to the webcam.

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

## Posture Detection Logic

The initial implementation uses the following experimental parameters:

| Parameter | Initial value | Purpose |
|---|---:|---|
| Landmark visibility threshold | 0.5 | Reject landmarks with low estimated visibility |
| Torso tilt threshold | 20 degrees | Initial threshold for detecting tilt |
| Warning duration | 1 second | Require sustained tilt before showing a warning |

These values are configurable in `posture_detection.py`.

The torso angle is calculated from the horizontal and vertical differences between the shoulder and hip midpoints:

```python
dx = shoulder_mid[0] - hip_mid[0]
dy = hip_mid[1] - shoulder_mid[1]

angle = math.degrees(math.atan2(dx, dy))
```

The angle is measured relative to the vertical direction in the image.

The warning logic is:

1. Detect the shoulder and hip landmarks.
2. Check whether the landmarks are sufficiently visible.
3. Calculate the torso angle.
4. Compare the absolute angle with the threshold.
5. Start a timer if the threshold is exceeded.
6. Display a warning if the tilt persists for the configured duration.
7. Reset the timer when the tilt falls below the threshold.

This is an initial rule-based approach, not a trained classifier. It can produce false positives because camera position, perspective, and natural body movement affect the measured angle.

## Troubleshooting

### MediaPipe cannot be imported

Run:

```powershell
python -m pip install mediapipe
```

Then verify:

```powershell
python -c "import mediapipe as mp; print(mp.__version__)"
```

If installation fails, check the complete error message and the Python version being used.

### OpenCV cannot open the webcam

Check that:

- The webcam is connected and enabled.
- No other application is exclusively using the camera.
- VS Code or the terminal has permission to access the camera.
- The correct camera index is being used.

The current code uses:

```python
cap = cv2.VideoCapture(0)
```

Some systems may require a different camera index.

### Model file not found

Make sure `pose_landmarker_full.task` is located in the project directory and that the terminal is running from that directory.

Check the current directory:

```powershell
Get-Location
```

List its contents:

```powershell
dir
```

### No body landmarks appear

Make sure the body is visible in the camera frame and that the lighting is sufficient. The shoulder and hip landmarks are particularly important for the current posture-detection algorithm.

### Unexpected posture warnings

The initial thresholds have not been validated for all camera positions. Adjust the threshold only after observing the angle during normal sitting and deliberate leaning.

## Current Limitations

- Posture detection currently uses simple geometric rules.
- The system has not been validated in a real vehicle.
- Forward slouching and sideways leaning are not yet comprehensively classified.
- Webcam perspective can affect the calculated angle.
- Facial behavior detection is not yet integrated into the current posture script.
- Recorded-video processing is planned but not yet documented as implemented.
- No driver-specific dataset has been used to train or evaluate a custom classifier.
- The current warnings are experimental and should not be treated as reliable safety alerts.

## Future Enhancements

The following features are planned for future development:

1. **Improved body posture classification** — Distinguish normal posture, sideways leaning, forward slouching, and other unwanted postures.
2. **Camera calibration** — Establish a baseline for normal posture based on the driver's position and camera placement.
3. **Facial landmark detection** — Analyze facial landmarks for eye closure and mouth opening.
4. **Drowsiness detection** — Explore Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), and temporal analysis.
5. **Temporal smoothing** — Reduce false alarms caused by brief movements or landmark jitter.
6. **Recorded-video support** — Analyze saved video files in addition to live webcam input.
7. **Dataset evaluation** — Evaluate detection accuracy using representative, appropriately labeled video samples.
8. **Integrated monitoring** — Combine body posture and facial behavior into a single driver-monitoring pipeline.
9. **Performance evaluation** — Measure detection latency, frame rate, and false-positive rates under different conditions.

## Disclaimer

This project is an educational computer vision prototype. It is intended for experimentation and research into body posture analysis.

The current system has not been validated for real-world driving or safety-critical use. Its warnings must not be relied upon to determine whether a driver is safe or unsafe to drive.

## Author

Developed as a Python-based computer vision project focused on driver posture monitoring, body landmark detection, and future extensions for facial behavior analysis.

/////   for running video //////

python driver_monitor.py --source "C:\Python\driver-unwanted-pose\test_video.mp4"