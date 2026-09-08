#!/usr/bin/env python3

import json
import cv2
import requests
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict
import numpy as np
from scipy.spatial.distance import cdist
from ultralytics import YOLO
from src.utils.game_logger import GameLogger
import evaluate

MODEL_PATH = 'models/finetuned_yolov8m.pt'
FINETUNED_RANKS = {'A', '2', '4', '6'}
API_URL = 'https://serverless.roboflow.com/playing-cards-ow27d/4'
API_KEY = 'ukokfyuNnrJZC3WdbtlW'
FRAME_SAMPLE = 5
CONFIDENCE_THRESHOLD = 0.5
VIDEO_DIR = Path('data/test_videos')
OUTPUT_DIR = Path('output')

CARD_VALUES = {
    'A': 11, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6,
    '7': 7, '8': 8, '9': 9, '10': 10,
    'J': 10, 'Q': 10, 'K': 10
}

PLAYER_ZONE = {'y_min': 1920, 'y_max': 3840}
DEALER_ZONE = {'y_min': 0, 'y_max': 1920}


@dataclass
class Detection:
    card_class: str
    confidence: float
    x: float
    y: float
    original_height: int = 3840

    @property
    def center(self):
        return (self.x, self.y)

    @property
    def scaled_y(self) -> float:
        return self.y * (self.original_height / 640)


@dataclass
class Track:
    track_id: int
    card_class: str
    detections: List[Detection] = field(default_factory=list)
    frames: List[int] = field(default_factory=list)
    first_frame: int = 0
    last_frame: int = 0

    def add_detection(self, detection: Detection, frame_idx: int):
        self.detections.append(detection)
        self.frames.append(frame_idx)
        self.last_frame = frame_idx
        if self.first_frame == 0:
            self.first_frame = frame_idx


class CardTracker:
    def __init__(self):
        self.next_track_id = 0
        self.tracks = {}
        self.frame_idx = 0

    def _deduplicate_detections(self, detections: List[Detection]) -> List[Detection]:
        if not detections:
            return detections

        seen_cards = {}
        deduplicated = []

        for det in detections:
            if det.card_class not in seen_cards:
                seen_cards[det.card_class] = det
                deduplicated.append(det)
            else:
                prev_det = seen_cards[det.card_class]
                y_distance = abs(det.y - prev_det.y)

                if y_distance < 300:
                    if det.confidence > prev_det.confidence:
                        deduplicated.remove(prev_det)
                        deduplicated.append(det)
                        seen_cards[det.card_class] = det
                else:
                    deduplicated.append(det)

        return deduplicated

    def update(self, detections: List[Detection]):
        detections = self._deduplicate_detections(detections)

        active_tracks = {tid: t for tid, t in self.tracks.items()}

        if not active_tracks or not detections:
            for det in detections:
                track = Track(self.next_track_id, det.card_class)
                track.add_detection(det, self.frame_idx)
                self.tracks[self.next_track_id] = track
                self.next_track_id += 1
        else:
            track_ids = list(active_tracks.keys())
            track_centers = np.array([active_tracks[tid].detections[-1].center for tid in track_ids])
            det_centers = np.array([det.center for det in detections])

            distances = cdist(track_centers, det_centers, metric='euclidean')

            matched_detections = set()
            matched_tracks = set()

            while True:
                min_dist = float('inf')
                min_i, min_j = -1, -1

                for i, tid in enumerate(track_ids):
                    if i in matched_tracks:
                        continue
                    for j, det in enumerate(detections):
                        if j in matched_detections:
                            continue
                        if distances[i, j] < min_dist:
                            min_dist = distances[i, j]
                            min_i, min_j = i, j

                if min_dist > 80 or min_i == -1:
                    break

                track_id = track_ids[min_i]
                self.tracks[track_id].add_detection(detections[min_j], self.frame_idx)
                matched_detections.add(min_j)
                matched_tracks.add(min_i)

            for j, det in enumerate(detections):
                if j not in matched_detections:
                    track = Track(self.next_track_id, det.card_class)
                    track.add_detection(det, self.frame_idx)
                    self.tracks[self.next_track_id] = track
                    self.next_track_id += 1

        self.frame_idx += 1


def extract_rank(card_class: str) -> str:
    if card_class.startswith('10'):
        return '10'
    return card_class[0] if card_class else ''


def calculate_hand_value(cards: List[str]) -> int:
    total = 0
    aces = 0

    for card_class in cards:
        rank = extract_rank(card_class)
        if rank == 'A':
            aces += 1
        total += CARD_VALUES.get(rank, 0)

    while total > 21 and aces > 0:
        total -= 10
        aces -= 1

    return total


def is_in_zone(y: float, zone: Dict) -> bool:
    return zone['y_min'] <= y <= zone['y_max']


def detect_finetuned(resized_frame, model) -> List[Detection]:
    detections = []
    try:
        results = model.predict(resized_frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        if results:
            for box in results[0].boxes:
                class_name = results[0].names.get(int(box.cls[0]), 'unknown')
                if extract_rank(class_name) in FINETUNED_RANKS:
                    detections.append(Detection(
                        card_class=class_name,
                        confidence=float(box.conf[0]),
                        x=float(box.xywh[0][0]),
                        y=float(box.xywh[0][1])
                    ))
    except Exception:
        pass
    return detections


def detect_roboflow(resized_frame, temp_path='/tmp/detection_frame.jpg') -> List[Detection]:
    detections = []
    try:
        cv2.imwrite(temp_path, resized_frame)
        with open(temp_path, 'rb') as img_file:
            response = requests.post(
                f'{API_URL}?api_key={API_KEY}',
                files={'file': img_file},
                timeout=10
            )
        if response.status_code == 200:
            for pred in response.json().get('predictions', []):
                class_name = pred.get('class', 'unknown')
                if extract_rank(class_name) not in FINETUNED_RANKS:
                    detections.append(Detection(
                        card_class=class_name,
                        confidence=pred.get('confidence', 0),
                        x=pred.get('x', 0),
                        y=pred.get('y', 0)
                    ))
    except Exception:
        pass
    return detections


def hybrid_detect(frame, model) -> List[Detection]:
    resized = cv2.resize(frame, (640, 640))
    return detect_finetuned(resized, model) + detect_roboflow(resized)


def build_track_summaries(tracker: CardTracker) -> List[Dict]:
    card_zone_tracks = {}
    for track in tracker.tracks.values():
        if not track.detections:
            continue

        scaled_y = track.detections[0].scaled_y
        if is_in_zone(scaled_y, PLAYER_ZONE):
            zone = 'player'
        elif is_in_zone(scaled_y, DEALER_ZONE):
            zone = 'dealer'
        else:
            continue

        key = (track.card_class, zone)
        card_zone_tracks.setdefault(key, []).append(track)

    summaries = []
    for (card_class, zone), tracks in card_zone_tracks.items():
        best_track = max(tracks, key=lambda t: len(t.detections))
        avg_confidence = sum(d.confidence for d in best_track.detections) / len(best_track.detections)
        y_positions = [d.scaled_y for d in best_track.detections]
        avg_y = sum(y_positions) / len(y_positions)
        y_range = max(y_positions) - min(y_positions) if len(y_positions) > 1 else 0

        summaries.append({
            'card': card_class,
            'zone': zone,
            'track': best_track,
            'avg_conf': avg_confidence,
            'first_frame': best_track.first_frame,
            'count': len(best_track.detections),
            'avg_y': avg_y,
            'y_stability': y_range
        })

    return summaries


def merge_sequential_tracks(summaries: List[Dict]) -> List[Dict]:
    merged = []
    used = set()

    for i, t1 in enumerate(summaries):
        if i in used:
            continue

        for j in range(i + 1, len(summaries)):
            if j in used:
                continue
            t2 = summaries[j]
            if (t1['card'] == t2['card'] and t1['zone'] == t2['zone']
                    and t2['first_frame'] - t1['first_frame'] < 10):
                if t2['count'] > t1['count']:
                    t1 = t2.copy()
                used.add(j)

        merged.append(t1)

    return merged


def filter_initial_deal(merged: List[Dict]) -> List[Dict]:
    initial_deal = []

    for t in merged:
        if t['first_frame'] >= 30:
            continue
        if t['count'] < 3:
            continue
        if t['y_stability'] > 250:
            continue

        quality_score = t['count'] * t['avg_conf'] / max(1, t['y_stability'] / 100)

        if t['y_stability'] < 100:
            min_quality = 0.8
        elif t['y_stability'] < 200:
            min_quality = 1.0
        else:
            min_quality = 1.2

        if quality_score < min_quality:
            continue
        if t['zone'] == 'player' and t['avg_y'] < 1500:
            continue
        if t['zone'] == 'dealer' and t['avg_y'] > 2200:
            continue

        initial_deal.append(t)

    return initial_deal


def select_final_tracks(summaries: List[Dict]) -> List[Dict]:
    merged = merge_sequential_tracks(summaries)
    initial_deal = filter_initial_deal(merged)
    later = [t for t in merged if t['first_frame'] >= 30]

    if len(initial_deal) < 4:
        dealer_later = [t for t in later if t['zone'] == 'dealer' and t['first_frame'] < 100]
        for t in dealer_later:
            if t not in initial_deal and len(initial_deal) < 4:
                if t['count'] >= 2 and t['y_stability'] <= 300:
                    initial_deal.append(t)
                    later.remove(t)

    initial_deal.sort(key=lambda x: (x['first_frame'], -x['count']))
    if len(initial_deal) > 4:
        initial_deal = initial_deal[:4]
    initial_deal.sort(key=lambda x: x['first_frame'])

    final_tracks = initial_deal + later
    final_tracks.sort(key=lambda x: x['first_frame'])
    return final_tracks


def determine_outcome(player_value: int, dealer_value: int) -> str:
    if player_value > 21:
        return 'DEALER WINS - Player Bust'
    if dealer_value > 21:
        return 'PLAYER WINS - Dealer Bust'
    if player_value > dealer_value:
        return 'PLAYER WINS'
    if dealer_value > player_value:
        return 'DEALER WINS'
    return 'PUSH'


def process_video(video_path: Path, model) -> Dict:
    print(f'\nProcessing: {video_path.name}')
    print('-' * 60)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tracker = CardTracker()
    game_logger = GameLogger(video_path.name, fps=fps)
    frame_idx = 0

    print('Step 1: Detection and tracking')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % FRAME_SAMPLE == 0:
            detections = hybrid_detect(frame, model)
            tracker.update(detections)

        frame_idx += 1

    cap.release()

    print('Step 2: Hand assignment and game logic')

    summaries = build_track_summaries(tracker)
    final_tracks = select_final_tracks(summaries)

    player_hand = []
    dealer_hand = []
    dealer_hidden_card = None
    card_timeline = []
    dealer_card_count = 0

    for t in final_tracks:
        card_class = t['card']
        zone = t['zone']
        track = t['track']

        card_timeline.append({
            'card': card_class,
            'zone': zone,
            'appeared_at_frame': track.first_frame,
            'sequence': len(card_timeline) + 1
        })

        event_type = 'initial_deal' if track.first_frame < 100 else 'hit'

        if zone == 'dealer':
            dealer_card_count += 1
            if dealer_card_count == 2 and track.first_frame < 100:
                dealer_hidden_card = card_class
                game_logger.add_card_event(card_class, 'dealer', track.first_frame,
                                           'initial_deal', visible=False)
            else:
                dealer_hand.append(card_class)
                game_logger.add_card_event(card_class, 'dealer', track.first_frame,
                                           event_type, visible=True)
        else:
            player_hand.append(card_class)
            game_logger.add_card_event(card_class, 'player', track.first_frame,
                                       event_type, visible=True)

    player_value = calculate_hand_value(player_hand)

    dealer_all_cards = dealer_hand.copy()
    if dealer_hidden_card:
        dealer_all_cards.append(dealer_hidden_card)
    dealer_value = calculate_hand_value(dealer_all_cards)

    outcome = determine_outcome(player_value, dealer_value)

    game_logger.set_final_state(player_hand, dealer_hand, dealer_hidden_card, outcome)
    log_file = OUTPUT_DIR / f'{video_path.stem}_game_log.json'
    game_logger.save_json(log_file)

    print(f'Player hand: {player_hand} (sum {player_value})')
    print(f'Dealer visible: {dealer_hand}')
    if dealer_hidden_card:
        print(f'Dealer hidden: {dealer_hidden_card}')
    print(f'Dealer sum (all cards): {dealer_value}')
    print(f'Outcome: {outcome}')

    return {
        'video': video_path.name,
        'fps': fps,
        'total_frames': total_frames,
        'player_cards': player_hand,
        'player_value': player_value,
        'dealer_visible_cards': dealer_hand,
        'dealer_hidden_card': dealer_hidden_card,
        'dealer_value': dealer_value,
        'outcome': outcome,
        'card_timeline': card_timeline,
        'game_log_file': str(log_file)
    }


def main():
    print('=' * 60)
    print('Blackjack Card Tracking Pipeline')
    print('Hybrid detection: fine-tuned model (A, 2, 4, 6) + Roboflow API (other ranks)')
    print('=' * 60)

    OUTPUT_DIR.mkdir(exist_ok=True)

    if Path(MODEL_PATH).exists():
        model = YOLO(MODEL_PATH)
        print(f'Loaded fine-tuned model: {MODEL_PATH}')
    else:
        model = YOLO('yolov8m.pt')
        print(f'Fine-tuned model not found, using pretrained yolov8m.pt as fallback')

    video_files = sorted(VIDEO_DIR.glob('*.MOV'))
    if not video_files:
        print(f'No videos found in {VIDEO_DIR}')
        return

    print(f'Found {len(video_files)} test videos')

    results = []
    for video_path in video_files:
        results.append(process_video(video_path, model))

    output_file = OUTPUT_DIR / 'pipeline_results.json'
    with open(output_file, 'w') as f:
        json.dump({'results': results}, f, indent=2)

    print('\n' + '=' * 60)
    print('SUMMARY')
    print('=' * 60)
    for r in results:
        print(f"{r['video']}: {r['outcome']} "
              f"(player {r['player_value']}, dealer {r['dealer_value']})")
    print(f'\nResults saved to: {output_file}')

    print('\n' + '=' * 60)
    print('EVALUATION')
    print('=' * 60)
    evaluate.run_evaluation(results)


if __name__ == '__main__':
    main()
