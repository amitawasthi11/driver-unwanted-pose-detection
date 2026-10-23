
import cv2
import mediapipe as mp
import math
import time

MODEL_PATH = "face_landmarker.task"

# Initial experimental values; calibrate for your camera.
MAR_THRESHOLD = 0.35
MOUTH_OPEN_DURATION = 1.0

# MediaPipe Face Landmarker indices
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
UPPER_LOWER_PAIRS = [
    (13, 14),
    (82, 87),
    (312, 317)
]

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


def distance(point_a, point_b):
    return math.hypot(
        point_a[0] - point_b[0],
        point_a[1] - point_b[1]
    )


def get_point(landmarks, index, width, height):
    point = landmarks[index]
    return (point.x * width, point.y * height)


def calculate_mar(landmarks, width, height):
    left = get_point(landmarks, MOUTH_LEFT, width, height)
    right = get_point(landmarks, MOUTH_RIGHT, width, height)

    horizontal = distance(left, right)

    if horizontal < 1e-6:
        return None

    vertical_distances = []

    for upper_index, lower_index in UPPER_LOWER_PAIRS:
        upper = get_point(
            landmarks, upper_index, width, height
        )
        lower = get_point(
            landmarks, lower_index, width, height
        )
        vertical_distances.append(distance(upper, lower))

    return sum(vertical_distances) / (
        len(vertical_distances) * horizontal
    )


options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_faces=1
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

start_time = time.monotonic()
mouth_open_start = None

print("Mouth opening detection started. Press Q to quit.")

try:
    with FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            success, frame = cap.read()

            if not success:
                print("Could not read webcam frame.")
                break

            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]

            rgb_frame = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

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
                mar = calculate_mar(face, width, height)

                if mar is not None:
                    cv2.putText(
                        frame,
                        f"MAR: {mar:.3f}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2
                    )

                    # Draw the landmarks used by the MAR calculation.
                    indices = [MOUTH_LEFT, MOUTH_RIGHT]
                    for pair in UPPER_LOWER_PAIRS:
                        indices.extend(pair)

                    for index in indices:
                        point = get_point(
                            face, index, width, height
                        )
                        cv2.circle(
                            frame,
                            (int(point[0]), int(point[1])),
                            3,
                            (0, 255, 0),
                            -1
                        )

                    if mar >= MAR_THRESHOLD:
                        if mouth_open_start is None:
                            mouth_open_start = time.monotonic()

                        duration = (
                            time.monotonic() - mouth_open_start
                        )

                        if duration >= MOUTH_OPEN_DURATION:
                            status = "Sustained mouth opening"
                            color = (0, 0, 255)
                        else:
                            status = "Mouth open - checking..."
                            color = (0, 165, 255)
                    else:
                        mouth_open_start = None
                        status = "Mouth appears closed"
                        color = (0, 255, 0)
                else:
                    mouth_open_start = None
                    status = "MAR unavailable"
                    color = (0, 165, 255)
            else:
                mouth_open_start = None

            cv2.putText(
                frame,
                status,
                (20, height - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

            cv2.imshow("Driver Mouth Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    cap.release()
    cv2.destroyAllWindows()
