
import cv2
import mediapipe as mp
import math
import time

MODEL_PATH = "pose_landmarker_full.task"
CAMERA_ID = 0

VISIBILITY_THRESHOLD = 0.5
LEAN_THRESHOLD = 20
ALERT_DURATION = 1.0
CALIBRATION_DURATION = 3.0

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

def midpoint(point_a, point_b):
    return (
        (point_a.x + point_b.x) / 2,
        (point_a.y + point_b.y) / 2
    )

def get_torso_angle(shoulder_mid, hip_mid):
    dx = shoulder_mid[0] - hip_mid[0]
    dy = hip_mid[1] - shoulder_mid[1]

    return math.degrees(math.atan2(dx, dy))

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO
)

cap = cv2.VideoCapture(CAMERA_ID)

if not cap.isOpened():
    raise RuntimeError(
        "Could not open webcam. Check camera permissions or CAMERA_ID."
    )

start_time = time.monotonic()
lean_start = None
baseline_angle = None
baseline_samples = []

print("Keep your upper body upright during the 3-second calibration.")
print("Press Q to quit.")

try:
    with PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            success, frame = cap.read()

            if not success:
                print("Could not read webcam frame.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            timestamp_ms = int(
                (time.monotonic() - start_time) * 1000
            )

            result = landmarker.detect_for_video(
                mp_image, timestamp_ms
            )

            elapsed = time.monotonic() - start_time
            status = "No pose detected"
            color = (0, 0, 255)

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]

                required_ids = [11, 12, 23, 24]
                reliable = all(
                    landmarks[i].visibility is not None
                    and landmarks[i].visibility >= VISIBILITY_THRESHOLD
                    for i in required_ids
                )

                if reliable:
                    left_shoulder = landmarks[11]
                    right_shoulder = landmarks[12]
                    left_hip = landmarks[23]
                    right_hip = landmarks[24]

                    shoulder_mid = midpoint(
                        left_shoulder, right_shoulder
                    )
                    hip_mid = midpoint(left_hip, right_hip)

                    angle = get_torso_angle(
                        shoulder_mid, hip_mid
                    )

                    for point in required_ids:
                        x = int(landmarks[point].x * frame.shape[1])
                        y = int(landmarks[point].y * frame.shape[0])
                        cv2.circle(
                            frame, (x, y), 6, (0, 255, 0), -1
                        )

                    cv2.line(
                        frame,
                        (
                            int(shoulder_mid[0] * frame.shape[1]),
                            int(shoulder_mid[1] * frame.shape[0])
                        ),
                        (
                            int(hip_mid[0] * frame.shape[1]),
                            int(hip_mid[1] * frame.shape[0])
                        ),
                        (255, 255, 0),
                        2
                    )

                    if elapsed < CALIBRATION_DURATION:
                        baseline_samples.append(angle)
                        status = "Calibrating - sit upright"
                        color = (0, 255, 255)
                    else:
                        if baseline_angle is None:
                            if baseline_samples:
                                baseline_angle = sum(
                                    baseline_samples
                                ) / len(baseline_samples)
                            else:
                                baseline_angle = angle

                        relative_angle = angle - baseline_angle

                        cv2.putText(
                            frame,
                            f"Torso angle: {angle:.1f}",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 255, 255), 2
                        )

                        cv2.putText(
                            frame,
                            f"Change: {relative_angle:.1f} deg",
                            (20, 75),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 255, 255), 2
                        )

                        if abs(relative_angle) >= LEAN_THRESHOLD:
                            if lean_start is None:
                                lean_start = time.monotonic()

                            duration = time.monotonic() - lean_start

                            if duration >= ALERT_DURATION:
                                status = "Possible sustained lean!"
                                color = (0, 0, 255)
                            else:
                                status = "Lean detected - checking..."
                                color = (0, 165, 255)
                        else:
                            lean_start = None
                            status = "Posture within threshold"
                            color = (0, 255, 0)
                else:
                    lean_start = None
                    status = "Landmarks unreliable"
                    color = (0, 165, 255)
            else:
                lean_start = None

            cv2.putText(
                frame,
                status,
                (20, frame.shape[0] - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, color, 2
            )

            cv2.imshow("Driver Posture Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    cap.release()
    cv2.destroyAllWindows()
