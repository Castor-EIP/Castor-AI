# modules/sport_ai/inference.py
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from pathlib import Path

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
BALL_CONFIDENCE = 0.45
SPORTS_BALL_CLASS_ID = 32
BORDER_COLOR = (0, 0, 255)
BORDER_THICKNESS = 6

MODEL_PATH = (Path(__file__).resolve().parents[2] / "models" / "yolov8n.pt")

def run(video_left: str, video_right: str, frameskip: int = 0):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA non disponible : PyTorch est installé en version CPU-only")

    cv2.destroyAllWindows()
    cv2.startWindowThread()

    torch.backends.cudnn.benchmark = True
    stream = torch.cuda.Stream()

    model = YOLO(str(MODEL_PATH))
    model.to("cuda")

    def detect_ball_fast(frame):
        img = cv2.resize(frame, (416, 416))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        with torch.cuda.stream(stream):
            results = model.predict(
                img, conf=0.25, imgsz=416, device=0,
                half=True, verbose=False, show=False
            )[0]

        scale_x = frame.shape[1] / 416
        scale_y = frame.shape[0] / 416
        ball_found = False

        for box in results.boxes:
            cls = int(box.cls)
            conf = float(box.conf)
            if cls == SPORTS_BALL_CLASS_ID and conf >= BALL_CONFIDENCE:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1 = int(x1 * scale_x); y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x); y2 = int(y2 * scale_y)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Ball {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                ball_found = True

        return frame, ball_found

    cap1 = cv2.VideoCapture(video_left)
    cap2 = cv2.VideoCapture(video_right)
    if not cap1.isOpened() or not cap2.isOpened():
        raise RuntimeError("Impossible d'ouvrir les vidéos")

    cv2.namedWindow("Videos", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Videos", WINDOW_WIDTH, WINDOW_HEIGHT)

    frame_count = 0
    active_cam = "None"

    while True:
        ret1, f1 = cap1.read()
        ret2, f2 = cap2.read()
        if not ret1 or not ret2:
            break

        if frame_count % (frameskip + 1) == 0:
            f1_ai, found1 = detect_ball_fast(f1)
            f2_ai, found2 = detect_ball_fast(f2)

            if found1:
                active_cam = "LEFT"
            elif found2:
                active_cam = "RIGHT"

            f1 = f1_ai
            f2 = f2_ai

        frame_count += 1

        left = cv2.resize(f1, (WINDOW_WIDTH // 2 - 5, WINDOW_HEIGHT))
        right = cv2.resize(f2, (WINDOW_WIDTH // 2 - 5, WINDOW_HEIGHT))

        bc_left, bc_right = (0, 0, 0), (0, 0, 0)
        if active_cam == "LEFT":
            bc_left = BORDER_COLOR
        elif active_cam == "RIGHT":
            bc_right = BORDER_COLOR

        left = cv2.copyMakeBorder(left, BORDER_THICKNESS, BORDER_THICKNESS, BORDER_THICKNESS,
                                  BORDER_THICKNESS, cv2.BORDER_CONSTANT, value=bc_left)
        right = cv2.copyMakeBorder(right, BORDER_THICKNESS, BORDER_THICKNESS, BORDER_THICKNESS,
                                   BORDER_THICKNESS, cv2.BORDER_CONSTANT, value=bc_right)

        left = cv2.resize(left, (WINDOW_WIDTH // 2 - 5, WINDOW_HEIGHT))
        right = cv2.resize(right, (WINDOW_WIDTH // 2 - 5, WINDOW_HEIGHT))

        spacer = np.zeros((WINDOW_HEIGHT, 10, 3), dtype=np.uint8)
        combined = np.hstack((left, spacer, right))
        cv2.imshow("Videos", combined)

        print(f"\rBall detected on: {active_cam}      ", end="")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()
