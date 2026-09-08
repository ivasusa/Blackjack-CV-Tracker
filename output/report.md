# Blackjack Card Tracking - Detection Report

Generated: 2026-09-08 20:50:46

Detection method: hybrid model.
Fine-tuned YOLOv8m handles ranks A, 2, 4, 6.
Roboflow pretrained model handles all other ranks.

## Per-Video Results

| Video | Expected player | Detected player | Expected dealer | Detected dealer | Expected outcome | Detected outcome | Cards OK | Outcome OK |
|---|---|---|---|---|---|---|---|---|
| player_21.MOV | 10H, AC | 10H | 8D, 9H | 8D, 9H | PLAYER WINS | DEALER WINS | NO | NO |
| player_bust.MOV | 10C, 8C, 7D | 10C, 8C, 7D | 10D | 10D | DEALER WINS - Player Bust | DEALER WINS - Player Bust | yes | yes |
| player_win.MOV | 10C, 8C | 10C, 8C | 10D | 10D | PLAYER WINS | PLAYER WINS | yes | yes |
| scenario1.MOV | 10S, 9D | 10S, 9D | 7H, JC | 7H, JC | PLAYER WINS | PLAYER WINS | yes | yes |
| scenario2.MOV | KH | KH, 9S | 6C, 10D, 8C | 8C, 10D | PLAYER WINS - Dealer Bust | PLAYER WINS | NO | NO |
| scenario3.MOV | 10C, 7H, 5H, 7S | 10C, 7H, 5H, 7S | 9C | 9C | DEALER WINS - Player Bust | DEALER WINS - Player Bust | yes | yes |
| scenario4.MOV | 9S, 10H | 9S, 10H | 9D, KC | 9D, KC | PUSH | PUSH | yes | yes |
| test.MOV | 8D, JD | 9D | 8H, JH | 2C | PUSH | PLAYER WINS | NO | NO |
| testA.MOV | QH, JH | QH, JH | 8C, KC | 8C, KC | PLAYER WINS | PLAYER WINS | yes | yes |

Videos processed but not evaluated (no ground truth recorded): equal.MOV, testB.MOV

## Detection Errors

- player_21.MOV: player: missed AC
- scenario2.MOV: player: spurious detection 9S; dealer: missed 6C
- test.MOV: player: expected 8D, detected 9D; player: missed JD; dealer: missed 8H; dealer: missed JH; dealer: spurious detection 2C

## Metrics

### Card-level (exact card class, player + dealer hands)

- True positives: 30
- False positives: 3
- False negatives: 6
- Precision: 0.909
- Recall: 0.833
- F1 score: 0.87

### Video-level

- Videos evaluated: 9
- Videos with all cards correct: 6 (67%)
- Videos with correct outcome: 6 (67%)

### Confusion matrix

Rank-level confusion matrix saved to: output/confusion_matrix.png

| expected \ detected | A | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | J | Q | K | missed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A |  |  |  |  |  |  |  |  |  |  |  |  |  | 1 |
| 5 |  |  |  |  | 1 |  |  |  |  |  |  |  |  |  |
| 6 |  |  |  |  |  |  |  |  |  |  |  |  |  | 1 |
| 7 |  |  |  |  |  |  | 4 |  |  |  |  |  |  |  |
| 8 |  |  |  |  |  |  |  | 5 | 1 |  |  |  |  | 1 |
| 9 |  |  |  |  |  |  |  |  | 5 |  |  |  |  |  |
| 10 |  |  |  |  |  |  |  |  |  | 9 |  |  |  |  |
| J |  |  |  |  |  |  |  |  |  |  | 2 |  |  | 2 |
| Q |  |  |  |  |  |  |  |  |  |  |  | 1 |  |  |
| K |  |  |  |  |  |  |  |  |  |  |  |  | 3 |  |
