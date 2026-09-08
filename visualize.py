#!/usr/bin/env python3

import sys
import json
import cv2
import requests
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = 'models/finetuned_yolov8m.pt'
FINETUNED_RANKS = {'A', '2', '4', '6'}
API_URL = 'https://serverless.roboflow.com/playing-cards-ow27d/4'
API_KEY = 'ukokfyuNnrJZC3WdbtlW'
FRAME_SAMPLE = 5
CONFIDENCE_THRESHOLD = 0.5
ZONE_BOUNDARY_Y = 1920
OUTPUT_HEIGHT = 1280
OUTPUT_FPS = 10

COLOR_FINETUNED = (90, 220, 110)
COLOR_ROBOFLOW = (40, 165, 255)
COLOR_ZONE = (200, 200, 200)
COLOR_PANEL = (30, 30, 30)
COLOR_TEXT = (255, 255, 255)
SUMMARY_HOLD_SECONDS = 4


def extract_rank(card_class: str) -> str:
    if card_class.startswith('10'):
        return '10'
    return card_class[0] if card_class else ''


def detect_frame(frame, model):
    resized = cv2.resize(frame, (640, 640))
    boxes = []

    try:
        results = model.predict(resized, conf=CONFIDENCE_THRESHOLD, verbose=False)
        if results:
            for box in results[0].boxes:
                class_name = results[0].names.get(int(box.cls[0]), 'unknown')
                if extract_rank(class_name) in FINETUNED_RANKS:
                    x, y, w, h = [float(v) for v in box.xywh[0]]
                    boxes.append({
                        'card': class_name,
                        'confidence': float(box.conf[0]),
                        'x': x, 'y': y, 'w': w, 'h': h,
                        'source': 'finetuned'
                    })
    except Exception:
        pass

    try:
        cv2.imwrite('/tmp/visualize_frame.jpg', resized)
        with open('/tmp/visualize_frame.jpg', 'rb') as img_file:
            response = requests.post(f'{API_URL}?api_key={API_KEY}',
                                     files={'file': img_file}, timeout=10)
        if response.status_code == 200:
            for pred in response.json().get('predictions', []):
                class_name = pred.get('class', 'unknown')
                if extract_rank(class_name) not in FINETUNED_RANKS:
                    boxes.append({
                        'card': class_name,
                        'confidence': pred.get('confidence', 0),
                        'x': pred.get('x', 0), 'y': pred.get('y', 0),
                        'w': pred.get('width', 0), 'h': pred.get('height', 0),
                        'source': 'roboflow'
                    })
    except Exception:
        pass

    return boxes


def draw_overlay(frame, boxes, frame_idx, fps, seen_player, seen_dealer):
    height, width = frame.shape[:2]
    scale_x = width / 640
    scale_y = height / 640

    cv2.line(frame, (0, ZONE_BOUNDARY_Y), (width, ZONE_BOUNDARY_Y), COLOR_ZONE, 6)
    cv2.putText(frame, 'DEALER ZONE', (40, ZONE_BOUNDARY_Y - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 2.2, COLOR_ZONE, 5)
    cv2.putText(frame, 'PLAYER ZONE', (40, ZONE_BOUNDARY_Y + 100),
                cv2.FONT_HERSHEY_SIMPLEX, 2.2, COLOR_ZONE, 5)

    for box in boxes:
        cx = box['x'] * scale_x
        cy = box['y'] * scale_y
        bw = box['w'] * scale_x
        bh = box['h'] * scale_y
        x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
        x2, y2 = int(cx + bw / 2), int(cy + bh / 2)

        color = COLOR_FINETUNED if box['source'] == 'finetuned' else COLOR_ROBOFLOW
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 8)

        label = f"{box['card']} {box['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 5)
        cv2.rectangle(frame, (x1, y1 - th - 24), (x1 + tw + 20, y1), color, -1)
        cv2.putText(frame, label, (x1 + 10, y1 - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 5)

        zone = 'player' if cy >= ZONE_BOUNDARY_Y else 'dealer'
        target = seen_player if zone == 'player' else seen_dealer
        if box['card'] not in target:
            target.append(box['card'])

    panel_height = 320
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, panel_height), COLOR_PANEL, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    timestamp = frame_idx / fps
    lines = [
        f'Frame {frame_idx}   t = {timestamp:.2f} s',
        f'Dealer: {", ".join(seen_dealer) if seen_dealer else "-"}',
        f'Player: {", ".join(seen_player) if seen_player else "-"}',
    ]
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (40, 80 + i * 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.2, COLOR_TEXT, 5)

    legend_y = height - 60
    cv2.putText(frame, 'green = fine-tuned model    orange = pretrained model',
                (40, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 1.8, COLOR_TEXT, 4)

    return frame


def card_value(card: str) -> int:
    rank = extract_rank(card)
    if rank == 'A':
        return 11
    if rank in ('J', 'Q', 'K', '10'):
        return 10
    return int(rank) if rank.isdigit() else 0


def hand_value(cards) -> int:
    total = sum(card_value(c) for c in cards)
    aces = sum(1 for c in cards if extract_rank(c) == 'A')
    while total > 21 and aces > 0:
        total -= 10
        aces -= 1
    return total


def build_summary_frame(width: int, height: int, log: dict):
    frame = cv2.UMat(height, width, cv2.CV_8UC3).get()
    frame[:] = (25, 25, 25)

    state = log['final_state']
    player = state['player_cards']
    dealer_visible = state['dealer_visible_cards']
    hidden = state['dealer_hidden_card']
    dealer_all = dealer_visible + ([hidden] if hidden else [])

    lines = [
        ('FINAL GAME STATE', 2.6, COLOR_TEXT),
        ('', 1.0, COLOR_TEXT),
        (f'Player: {", ".join(player) if player else "-"}', 2.0, COLOR_TEXT),
        (f'Player sum: {hand_value(player)}', 2.0, COLOR_TEXT),
        ('', 1.0, COLOR_TEXT),
        (f'Dealer visible: {", ".join(dealer_visible) if dealer_visible else "-"}',
         2.0, COLOR_TEXT),
        (f'Dealer hidden: {hidden if hidden else "-"}', 2.0, COLOR_TEXT),
        (f'Dealer sum: {hand_value(dealer_all)}', 2.0, COLOR_TEXT),
        ('', 1.0, COLOR_TEXT),
        (f'Outcome: {state["outcome"]}', 2.4, COLOR_FINETUNED),
    ]

    block_height = sum(int(scale * 55) + 40 for _, scale, _ in lines)
    y = max(200, (height - block_height) // 2)
    for text, scale, color in lines:
        if text:
            cv2.putText(frame, text, (80, y), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, color, 5)
        y += int(scale * 55) + 40

    return frame


def visualize(video_path: Path, output_path: Path, model):
    print(f'Processing: {video_path.name}')

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detections_by_frame = {}
    frame_idx = 0
    print('Step 1: Running detection on sampled frames')

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % FRAME_SAMPLE == 0:
            detections_by_frame[frame_idx] = detect_frame(frame, model)
        frame_idx += 1

    cap.release()
    print(f'Detected on {len(detections_by_frame)} sampled frames')

    out_width = int(width * OUTPUT_HEIGHT / height)
    out_height = OUTPUT_HEIGHT
    if out_width % 2 == 1:
        out_width += 1

    writer = cv2.VideoWriter(str(output_path),
                             cv2.VideoWriter_fourcc(*'avc1'),
                             OUTPUT_FPS, (out_width, out_height))

    cap = cv2.VideoCapture(str(video_path))
    frame_idx = 0
    last_boxes = []
    seen_player = []
    seen_dealer = []
    print('Step 2: Rendering annotated video')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in detections_by_frame:
            last_boxes = detections_by_frame[frame_idx]

        annotated = draw_overlay(frame, last_boxes, frame_idx, fps,
                                 seen_player, seen_dealer)
        writer.write(cv2.resize(annotated, (out_width, out_height)))
        frame_idx += 1

    cap.release()

    log_file = Path('output') / f'{video_path.stem}_game_log.json'
    if log_file.exists():
        with open(log_file) as f:
            log = json.load(f)
        summary = build_summary_frame(width, height, log)
        summary_resized = cv2.resize(summary, (out_width, out_height))
        for _ in range(SUMMARY_HOLD_SECONDS * OUTPUT_FPS):
            writer.write(summary_resized)

    writer.release()
    print(f'Annotated video saved: {output_path}')


def main():
    videos = sys.argv[1:] if len(sys.argv) > 1 else ['scenario1.MOV']

    output_dir = Path('output/demo')
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(MODEL_PATH) if Path(MODEL_PATH).exists() else YOLO('yolov8m.pt')

    for name in videos:
        video_path = Path('data/test_videos') / name
        if not video_path.exists():
            print(f'Video not found: {video_path}')
            continue
        output_path = output_dir / f'{video_path.stem}_annotated.mp4'
        visualize(video_path, output_path, model)


if __name__ == '__main__':
    main()
