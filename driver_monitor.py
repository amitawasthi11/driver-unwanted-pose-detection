import argparse
import csv
import math
import os
import time
from datetime import datetime

import cv2
import mediapipe as mp

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

POSE_MODEL = "pose_landmarker_full.task"
FACE_MODEL = "face_landmarker.task"
CAMERA_ID = 0

LOG_FILE = "driver_events.csv"
CSV_COLUMNS = [
    "timestamp",
    "event_type",
    "ear",
    "mar",
    "posture_angle_deg",
    "data_status",
]

VISIBILITY_THRESHOLD = 0.5
LEAN_THRESHOLD = 20.0
EAR_THRESHOLD = 0.20
MAR_THRESHOLD = 0.35

LEAN_DURATION = 1.0
EYE_CLOSED_DURATION = 1.5
MOUTH_OPEN_DURATION = 1.0
CALIBRATION_DURATION = 3.0

# Pose landmarks: shoulders and hips
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24

# Face Landmarker eye landmarks
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Face Landmarker mouth landmarks
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
MOUTH_PAIRS = [(13, 14), (82, 87), (312, 317)]


# --------------------------------------------------
# CSV LOGGING
# --------------------------------------------------

def initialize_csv():
    """Create the CSV file and header if it does not exist or is empty."""
    if not os.path.exists(LOG_FILE) or os.path.getsize(LOG_FILE) == 0:
        with open(LOG_FILE, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
            writer.writeheader()


def log_event(event_type, ear=None, mar=None, posture_angle=None,
              data_status="OK"):
    """Append one event and the available measurements to the CSV file."""
    timestamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
    row = {
        "timestamp": timestamp,
        "event_type": event_type,
        "ear": round(float(ear), 4) if is_valid_measurement(ear) else "",
        "mar": round(float(mar), 4) if is_valid_measurement(mar) else "",
        "posture_angle_deg": (
            round(float(posture_angle), 2)
            if is_valid_measurement(posture_angle) else ""
        ),
        "data_status": data_status,
    }
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writerow(row)


def is_valid_measurement(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


# --------------------------------------------------
# COMMON HELPERS
# --------------------------------------------------

def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def midpoint(a, b):
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def face_point(landmarks, index, width, height):
    point = landmarks[index]
    return (point.x * width, point.y * height)


def draw_line(frame, text, y, color=(255, 255, 255)):
    cv2.putText(
        frame,
        text,
        (15, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        color,
        2,
    )


# --------------------------------------------------
# POSTURE DETECTION
# --------------------------------------------------

def get_torso_angle(landmarks):
    required = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]

    for index in required:
        visibility = getattr(landmarks[index], "visibility", 0.0)
        if visibility is None or visibility < VISIBILITY_THRESHOLD:
            return None

    left_shoulder = landmarks[LEFT_SHOULDER]
    right_shoulder = landmarks[RIGHT_SHOULDER]
    left_hip = landmarks[LEFT_HIP]
    right_hip = landmarks[RIGHT_HIP]

    shoulder_mid = midpoint(
        (left_shoulder.x, left_shoulder.y),
        (right_shoulder.x, right_shoulder.y),
    )
    hip_mid = midpoint(
        (left_hip.x, left_hip.y),
        (right_hip.x, right_hip.y),
    )

    dx = shoulder_mid[0] - hip_mid[0]
    dy = hip_mid[1] - shoulder_mid[1]
    return math.degrees(math.atan2(dx, dy))


# --------------------------------------------------
# EYE ASPECT RATIO (EAR)
# --------------------------------------------------

def calculate_ear(landmarks, indices, width, height):
    points = [face_point(landmarks, index, width, height) for index in indices]
    horizontal = distance(points[0], points[3])
    if horizontal < 1e-6:
        return None

    vertical_1 = distance(points[1], points[5])
    vertical_2 = distance(points[2], points[4])
    return (vertical_1 + vertical_2) / (2 * horizontal)


# --------------------------------------------------
# MOUTH ASPECT RATIO (MAR)
# --------------------------------------------------

def calculate_mar(landmarks, width, height):
    left = face_point(landmarks, MOUTH_LEFT, width, height)
    right = face_point(landmarks, MOUTH_RIGHT, width, height)
    horizontal = distance(left, right)
    if horizontal < 1e-6:
        return None

    vertical_distances = []
    for upper_index, lower_index in MOUTH_PAIRS:
        upper = face_point(landmarks, upper_index, width, height)
        lower = face_point(landmarks, lower_index, width, height)
        vertical_distances.append(distance(upper, lower))

    return sum(vertical_distances) / (len(vertical_distances) * horizontal)


# --------------------------------------------------
# MAIN APPLICATION
# --------------------------------------------------

def main():
    global LOG_FILE

    parser = argparse.ArgumentParser(description="Driver Unwanted Pose Detection")
    parser.add_argument(
        "--source",
        default=str(CAMERA_ID),
        help="Webcam index (0) or path to a recorded video",
    )
    parser.add_argument(
        "--log",
        default=LOG_FILE,
        help="CSV output path (default: driver_events.csv)",
    )
    args = parser.parse_args()
    LOG_FILE = args.log
    initialize_csv()

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    is_webcam = isinstance(source, int)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if not video_fps or not math.isfinite(video_fps) or video_fps <= 0:
        video_fps = 30.0
    playback_delay = 1 if is_webcam else max(1, round(1000 / video_fps))

    BaseOptions = mp.tasks.BaseOptions
    RunningMode = mp.tasks.vision.RunningMode
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseOptions = mp.tasks.vision.PoseLandmarkerOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceOptions = mp.tasks.vision.FaceLandmarkerOptions

    pose_options = PoseOptions(
        base_options=BaseOptions(model_asset_path=POSE_MODEL),
        running_mode=RunningMode.VIDEO,
    )
    face_options = FaceOptions(
        base_options=BaseOptions(model_asset_path=FACE_MODEL),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
    )

    print("Source:", "Webcam" if is_webcam else source)
    print("CSV log:", os.path.abspath(LOG_FILE))
    print("Press Q to quit.")
    print("Keep your upper body upright during initial calibration.")

    start_time = time.monotonic()
    last_timestamp_ms = -1

    lean_start = None
    eye_closed_start = None
    mouth_open_start = None
    baseline_samples = []
    baseline_angle = None

    lean_logged = False
    eye_closed_logged = False
    mouth_open_logged = False
    previous_missing_status = None

    try:
        with PoseLandmarker.create_from_options(pose_options) as pose_model, \
             FaceLandmarker.create_from_options(face_options) as face_model:
            while True:
                success, frame = cap.read()
                if not success:
                    print("Video ended or could not read frame.")
                    break

                # Mirror the webcam view. This is also applied to recorded video.
                frame = cv2.flip(frame, 1)
                height, width = frame.shape[:2]
                now = time.monotonic()
                elapsed = now - start_time

                # Measurements are reset for every frame.
                ear = None
                mar = None
                angle = None
                posture_event_pending = False

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                # VIDEO mode requires strictly increasing timestamps.
                timestamp_ms = max(last_timestamp_ms + 1, int(elapsed * 1000))
                last_timestamp_ms = timestamp_ms
                pose_result = pose_model.detect_for_video(image, timestamp_ms)
                face_result = face_model.detect_for_video(image, timestamp_ms)

                posture_status = "Pose not detected"
                eye_status = "Face/eyes not detected"
                mouth_status = "Face/mouth not detected"
                posture_color = (0, 0, 255)
                eye_color = (0, 0, 255)
                mouth_color = (0, 0, 255)

                # --------------------------------------
                # POSTURE
                # --------------------------------------
                if pose_result.pose_landmarks:
                    pose = pose_result.pose_landmarks[0]
                    angle = get_torso_angle(pose)

                    if angle is None:
                        lean_start = None
                        lean_logged = False
                        posture_status = "Pose landmarks unreliable"
                        posture_color = (0, 165, 255)
                    elif elapsed < CALIBRATION_DURATION:
                        baseline_samples.append(angle)
                        lean_start = None
                        lean_logged = False
                        posture_status = "Calibrating: sit upright"
                        posture_color = (0, 255, 255)
                    else:
                        if baseline_angle is None:
                            baseline_angle = (
                                sum(baseline_samples) / len(baseline_samples)
                                if baseline_samples else angle
                            )

                        relative_angle = angle - baseline_angle
                        if abs(relative_angle) >= LEAN_THRESHOLD:
                            if lean_start is None:
                                lean_start = now

                            if now - lean_start >= LEAN_DURATION:
                                posture_status = (
                                    f"ALERT: sustained lean ({relative_angle:.1f} deg)"
                                )
                                posture_color = (0, 0, 255)
                                if not lean_logged:
                                    posture_event_pending = True
                                    lean_logged = True
                            else:
                                posture_status = f"Leaning: {relative_angle:.1f} deg"
                                posture_color = (0, 165, 255)
                        else:
                            lean_start = None
                            lean_logged = False
                            posture_status = f"Posture OK ({relative_angle:.1f} deg)"
                            posture_color = (0, 255, 0)
                else:
                    lean_start = None
                    lean_logged = False
                    posture_status = "Pose not detected"
                    posture_color = (0, 165, 255)

                # --------------------------------------
                # EYES + MOUTH
                # --------------------------------------
                if face_result.face_landmarks:
                    face = face_result.face_landmarks[0]
                    left_ear = calculate_ear(face, LEFT_EYE, width, height)
                    right_ear = calculate_ear(face, RIGHT_EYE, width, height)
                    mar = calculate_mar(face, width, height)

                    # Eye closure
                    if left_ear is not None and right_ear is not None:
                        ear = (left_ear + right_ear) / 2
                        if ear < EAR_THRESHOLD:
                            if eye_closed_start is None:
                                eye_closed_start = now
                            if now - eye_closed_start >= EYE_CLOSED_DURATION:
                                eye_status = f"ALERT: prolonged eye closure (EAR {ear:.3f})"
                                eye_color = (0, 0, 255)
                                if not eye_closed_logged:
                                    log_event("EYE_CLOSED", ear, mar, angle)
                                    eye_closed_logged = True
                            else:
                                eye_status = f"Eyes closing (EAR {ear:.3f})"
                                eye_color = (0, 165, 255)
                        else:
                            eye_closed_start = None
                            eye_closed_logged = False
                            eye_status = f"Eyes open (EAR {ear:.3f})"
                            eye_color = (0, 255, 0)
                    else:
                        eye_closed_start = None
                        eye_closed_logged = False
                        eye_status = "EAR unavailable"
                        eye_color = (0, 165, 255)

                    # Mouth opening (can indicate talking, laughing, or yawning)
                    if mar is not None:
                        if mar >= MAR_THRESHOLD:
                            if mouth_open_start is None:
                                mouth_open_start = now
                            if now - mouth_open_start >= MOUTH_OPEN_DURATION:
                                mouth_status = f"Mouth open (possible yawn) MAR {mar:.3f}"
                                mouth_color = (0, 165, 255)
                                if not mouth_open_logged:
                                    log_event("MOUTH_OPEN", ear, mar, angle)
                                    mouth_open_logged = True
                            else:
                                mouth_status = f"Mouth open (MAR {mar:.3f})"
                                mouth_color = (0, 165, 255)
                        else:
                            mouth_open_start = None
                            mouth_open_logged = False
                            mouth_status = f"Mouth OK (MAR {mar:.3f})"
                            mouth_color = (0, 255, 0)
                    else:
                        mouth_open_start = None
                        mouth_open_logged = False
                        mouth_status = "MAR unavailable"
                        mouth_color = (0, 165, 255)
                else:
                    eye_closed_start = None
                    mouth_open_start = None
                    eye_closed_logged = False
                    mouth_open_logged = False

                # Log posture after face processing so all available measurements
                # belong to this frame.
                if posture_event_pending:
                    log_event("POSTURE_LEAN", ear, mar, angle)

                # Log changes in which measurements are unavailable, not every frame.
                missing = []
                if not is_valid_measurement(ear):
                    missing.append("EAR")
                if not is_valid_measurement(mar):
                    missing.append("MAR")
                if not is_valid_measurement(angle):
                    missing.append("POSTURE_ANGLE")
                missing_status = "|".join(missing) if missing else "OK"

                if missing_status != previous_missing_status:
                    if missing:
                        log_event(
                            "DATA_MISSING",
                            ear=ear,
                            mar=mar,
                            posture_angle=angle,
                            data_status=missing_status,
                        )
                    previous_missing_status = missing_status

                # --------------------------------------
                # DISPLAY
                # --------------------------------------
                draw_line(frame, "DRIVER MONITOR - EXPERIMENTAL", 30, (255, 255, 255))
                draw_line(frame, posture_status, 65, posture_color)
                draw_line(frame, eye_status, 100, eye_color)
                draw_line(frame, mouth_status, 135, mouth_color)
                draw_line(frame, "Press Q to quit", height - 20, (255, 255, 255))
                cv2.imshow("Driver Unwanted Pose Detection", frame)

                if cv2.waitKey(playback_delay) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
