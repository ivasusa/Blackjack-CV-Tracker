# Blackjack Card Tracking System

Bachelor thesis project. The system automatically tracks the state of a
Blackjack game from a video recorded with a fixed iPhone camera above the
table. It detects cards, assigns them to the player or dealer hand based on
table zones, calculates hand values, handles the hidden dealer card, and
writes a complete game log in JSON format.

## Project Structure

```
diplomski/
├── pipeline.py              Main pipeline (detection, tracking, game logic)
├── evaluate.py              Evaluation against ground truth (metrics, report)
├── ground_truth.json        Expected cards and outcome per test video
├── GROUND_TRUTH.md          Human-readable version of the ground truth
├── requirements.txt
├── src/
│   └── utils/
│       └── game_logger.py   Game state logging to JSON
├── models/
│   └── finetuned_yolov8m.pt Fine-tuned YOLOv8m model (ranks A, 2, 4, 6)
├── data/
│   ├── test_videos/         Test recordings (.MOV)
│   ├── finetuning/          Images and labels used for fine-tuning
│   └── dataset_finetuning.yaml
└── output/                  Generated on each run
    ├── <video>_game_log.json
    ├── pipeline_results.json
    ├── evaluation.json
    ├── report.md            Detection report for the whole run
    └── confusion_matrix.png
```

## How It Works

1. Frames are sampled from the video (every 5th frame).
2. Each sampled frame goes through hybrid card detection:
   - a fine-tuned YOLOv8m model detects ranks A, 2, 4 and 6 (these ranks
     were unreliable with the pretrained model, so the model was fine-tuned
     on 64 custom images taken from the same camera setup),
   - the pretrained Roboflow playing-cards model detects all other ranks.
3. Detections are tracked across frames (centroid matching) and filtered
   (minimum number of detections, position stability, confidence) to remove
   flickering and false positives.
4. Each stable card is assigned to the player or dealer using fixed table
   zones (the camera position is fixed, so the zones are constant).
5. The second dealer card in the initial deal is treated as the hidden card.
6. Hand values are calculated using Blackjack rules (Ace counts as 11 or 1).
7. The outcome is determined and everything is written to the game log.

## Usage

```bash
pip install -r requirements.txt
python3 pipeline.py
```

Each run processes all videos in `data/test_videos/` and generates a full
detection report in `output/report.md`, including per-video results,
detection errors, precision/recall/F1, outcome accuracy and a rank-level
confusion matrix (`output/confusion_matrix.png`).

To re-run only the evaluation on existing results:

```bash
python3 evaluate.py
```

## Ground Truth

Expected cards and outcomes for every test video are recorded in
`ground_truth.json` (see also `GROUND_TRUTH.md`). Videos without recorded
ground truth are still processed but skipped in the metrics.
