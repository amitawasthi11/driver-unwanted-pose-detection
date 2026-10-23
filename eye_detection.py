
import cv2
import mediapipe as mp
import math
import time

MODEL_PATH = "face_landmarker.task"

# Starting values only; calibrate using your own webcam.
EAR_THRESHOLD = 0.20
CLOSED_DURATION = 1.5

# Six landmarks per eye:
# [outer corner, upper eyelid, upper eyelid,
#  inner corner, lower eyelid, lower eyelid]
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


def distance(point_a, point_b):
    return math.hypot(
        point_a[0] - point_b[0],
        point_a[1] - point_b[1]
    )


def calculate_ear(landmarks, eye_indices, width, height):
    points = []

    for index in eye_indices:
        landmark = landmarks[index]
        points.append((
            landmark.x * width,
            landmark.y * height
        ))

    horizontal = distance(points[0], points[3])

    if horizontal < 1e-6:
        return None

    vertical_1 = distance(points[1], points[5])
    vertical_2 = distance(points[2], points[4])

    return (vertical_1 + vertical_2) / (2 * horizontal)


options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_faces=1
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

start_time = time.monotonic()
closed_start = None

print("Eye closure detection started. Press Q to quit.")

try:
    with FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            success, frame = cap.read()

            if not success:
                print("Could not read webcam frame.")
                break

            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]

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

            status = "No face detected"
            color = (0, 0, 255)

            if result.face_landmarks:
                face = result.face_landmarks[0]

                left_ear = calculate_ear(
                    face, LEFT_EYE, width, height
                )
                right_ear = calculate_ear(
                    face, RIGHT_EYE, width, height
                )

                if left_ear is not None and right_ear is not None:
                    average_ear = (left_ear + right_ear) / 2

                    cv2.putText(
                        frame,
                        f"Left EAR: {left_ear:.3f}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (255, 255, 255), 2
                    )

                    cv2.putText(
                        frame,
                        f"Right EAR: {right_ear:.3f}",
                        (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (255, 255, 255), 2
                    )

                    # Draw eye landmarks.
                    for index in LEFT_EYE + RIGHT_EYE:
                        point = face[index]
                        x = int(point.x * width)
                        y = int(point.y * height)
                        cv2.circle(
                            frame, (x, y), 3, (0, 255, 0), -1
                        )

                    if average_ear < EAR_THRESHOLD:
                        if closed_start is None:
                            closed_start = time.monotonic()

                        duration = time.monotonic() - closed_start

                        if duration >= CLOSED_DURATION:
                            status = "Prolonged eye closure!"
                            color = (0, 0, 255)
                        else:
                            status = "Eyes possibly closed..."
                            color = (0, 165, 255)
                    else:
                        closed_start = None
                        status = "Eyes appear open"
                        color = (0, 255, 0)
                else:
                    closed_start = None
                    status = "EAR unavailable"
                    color = (0, 165, 255)
            else:
                closed_start = None

            cv2.putText(
                frame,
                status,
                (20, height - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, color, 2
            )

            cv2.imshow("Driver Eye Closure Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    cap.release()
    cv2.destroyAllWindows()
