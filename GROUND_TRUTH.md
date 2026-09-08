# Ground Truth: Expected Detections Per Video

Machine-readable version used by the evaluation script: `ground_truth.json`.
Dealer cards are listed as a set (visible + hidden), since the hole card is
one of them and is revealed later in the video.

## Test Videos and Expected Results

### test.MOV
- Player cards: 8D, JD (sum 18)
- Dealer cards: 8H, JH (sum 18)
- Expected outcome: PUSH

### testA.MOV
- Player cards: QH, JH (sum 20)
- Dealer cards: 8C, KC (sum 18)
- Expected outcome: PLAYER WINS

### player_bust.MOV
- Player cards: 10C, 8C, 7D (sum 25, bust)
- Dealer cards: 10D
- Expected outcome: DEALER WINS - Player Bust

### player_win.MOV
- Player cards: 10C, 8C (sum 18)
- Dealer cards: 10D
- Expected outcome: PLAYER WINS

### player_21.MOV
- Player cards: 10H, AC (sum 21)
- Dealer cards: 8D, 9H (sum 17)
- Expected outcome: PLAYER WINS

### scenario1.MOV
- Player cards: 10S, 9D (sum 19)
- Dealer cards: 7H, JC (sum 17)
- Expected outcome: PLAYER WINS

### scenario2.MOV
- Player cards: KH (sum 10)
- Dealer cards: 6C, 10D, 8C (sum 24, bust)
- Expected outcome: PLAYER WINS - Dealer Bust

### scenario3.MOV
- Player cards: 10C, 7H, 5H, 7S (sum 29, bust)
- Dealer cards: 9C
- Expected outcome: DEALER WINS - Player Bust

### scenario4.MOV
- Player cards: 9S, 10H (sum 19)
- Dealer cards: 9D, KC (sum 19)
- Expected outcome: PUSH

## Videos Without Recorded Ground Truth

- equal.MOV
- testB.MOV

These are still processed by the pipeline, but they are skipped in the
evaluation metrics until their expected values are added to `ground_truth.json`.
