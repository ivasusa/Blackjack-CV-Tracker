#!/usr/bin/env python3

import json
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

GROUND_TRUTH_FILE = Path('ground_truth.json')
OUTPUT_DIR = Path('output')
RANKS = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']


def extract_rank(card: str) -> str:
    if card.startswith('10'):
        return '10'
    return card[0] if card else ''


def extract_suit(card: str) -> str:
    if card.startswith('10'):
        return card[2] if len(card) > 2 else ''
    return card[1] if len(card) > 1 else ''


def load_ground_truth() -> dict:
    if not GROUND_TRUTH_FILE.exists():
        return {}
    with open(GROUND_TRUTH_FILE) as f:
        return json.load(f)


def match_cards(expected: list, detected: list) -> dict:
    expected_left = list(expected)
    detected_left = list(detected)
    true_positives = []
    confusions = []

    for card in list(expected_left):
        if card in detected_left:
            true_positives.append(card)
            expected_left.remove(card)
            detected_left.remove(card)

    for exp_card in list(expected_left):
        exp_suit = extract_suit(exp_card)
        for det_card in list(detected_left):
            if extract_suit(det_card) == exp_suit:
                confusions.append((exp_card, det_card))
                expected_left.remove(exp_card)
                detected_left.remove(det_card)
                break

    return {
        'true_positives': true_positives,
        'confusions': confusions,
        'missed': expected_left,
        'spurious': detected_left
    }


def evaluate_video(result: dict, gt: dict) -> dict:
    detected_player = result['player_cards']
    detected_dealer = result['dealer_visible_cards'].copy()
    if result.get('dealer_hidden_card'):
        detected_dealer.append(result['dealer_hidden_card'])

    player_match = match_cards(gt['player_cards'], detected_player)
    dealer_match = match_cards(gt['dealer_cards'], detected_dealer)

    all_cards_correct = (
        not player_match['confusions'] and not player_match['missed']
        and not player_match['spurious']
        and not dealer_match['confusions'] and not dealer_match['missed']
        and not dealer_match['spurious']
    )
    outcome_correct = result['outcome'] == gt['outcome']

    return {
        'video': result['video'],
        'expected_player': gt['player_cards'],
        'detected_player': detected_player,
        'expected_dealer': gt['dealer_cards'],
        'detected_dealer': detected_dealer,
        'expected_outcome': gt['outcome'],
        'detected_outcome': result['outcome'],
        'player_match': player_match,
        'dealer_match': dealer_match,
        'all_cards_correct': all_cards_correct,
        'outcome_correct': outcome_correct
    }


def compute_metrics(evaluations: list) -> dict:
    tp = fp = fn = 0
    confusion_pairs = []

    for ev in evaluations:
        for match in (ev['player_match'], ev['dealer_match']):
            tp += len(match['true_positives'])
            fp += len(match['spurious']) + len(match['confusions'])
            fn += len(match['missed']) + len(match['confusions'])
            confusion_pairs.extend(match['confusions'])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    outcome_correct = sum(1 for ev in evaluations if ev['outcome_correct'])
    cards_correct = sum(1 for ev in evaluations if ev['all_cards_correct'])

    return {
        'card_level': {
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'precision': round(precision, 3),
            'recall': round(recall, 3),
            'f1_score': round(f1, 3)
        },
        'video_level': {
            'videos_evaluated': len(evaluations),
            'outcome_correct': outcome_correct,
            'outcome_accuracy': round(outcome_correct / len(evaluations), 3) if evaluations else 0.0,
            'all_cards_correct': cards_correct,
            'card_accuracy': round(cards_correct / len(evaluations), 3) if evaluations else 0.0
        },
        'confusion_pairs': [{'expected': e, 'detected': d} for e, d in confusion_pairs]
    }


def build_confusion_matrix(evaluations: list) -> np.ndarray:
    labels = RANKS + ['missed']
    matrix = np.zeros((len(RANKS), len(labels)), dtype=int)

    for ev in evaluations:
        for match in (ev['player_match'], ev['dealer_match']):
            for card in match['true_positives']:
                r = RANKS.index(extract_rank(card))
                matrix[r][r] += 1
            for exp_card, det_card in match['confusions']:
                r = RANKS.index(extract_rank(exp_card))
                c = RANKS.index(extract_rank(det_card))
                matrix[r][c] += 1
            for card in match['missed']:
                r = RANKS.index(extract_rank(card))
                matrix[r][len(RANKS)] += 1

    return matrix


def save_confusion_matrix_plot(matrix: np.ndarray, output_path: Path):
    labels = RANKS + ['missed']
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap='Blues')

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(len(RANKS)))
    ax.set_yticklabels(RANKS)
    ax.set_xlabel('Detected rank')
    ax.set_ylabel('Expected rank')
    ax.set_title('Rank-level Confusion Matrix (all evaluated videos)')

    for i in range(len(RANKS)):
        for j in range(len(labels)):
            if matrix[i][j] > 0:
                ax.text(j, i, str(matrix[i][j]), ha='center', va='center',
                        color='white' if matrix[i][j] > matrix.max() / 2 else 'black')

    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def format_cards(cards: list) -> str:
    return ', '.join(cards) if cards else '-'


def write_report(evaluations: list, metrics: dict, skipped: list, output_path: Path):
    lines = []
    lines.append('# Blackjack Card Tracking - Detection Report')
    lines.append('')
    lines.append(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    lines.append('')
    lines.append('Detection method: hybrid model.')
    lines.append('Fine-tuned YOLOv8m handles ranks A, 2, 4, 6.')
    lines.append('Roboflow pretrained model handles all other ranks.')
    lines.append('')

    lines.append('## Per-Video Results')
    lines.append('')
    lines.append('| Video | Expected player | Detected player | Expected dealer | Detected dealer | Expected outcome | Detected outcome | Cards OK | Outcome OK |')
    lines.append('|---|---|---|---|---|---|---|---|---|')
    for ev in evaluations:
        lines.append(
            f"| {ev['video']} "
            f"| {format_cards(ev['expected_player'])} "
            f"| {format_cards(ev['detected_player'])} "
            f"| {format_cards(ev['expected_dealer'])} "
            f"| {format_cards(ev['detected_dealer'])} "
            f"| {ev['expected_outcome']} "
            f"| {ev['detected_outcome']} "
            f"| {'yes' if ev['all_cards_correct'] else 'NO'} "
            f"| {'yes' if ev['outcome_correct'] else 'NO'} |"
        )
    lines.append('')

    if skipped:
        lines.append('Videos processed but not evaluated (no ground truth recorded): '
                     + ', '.join(skipped))
        lines.append('')

    lines.append('## Detection Errors')
    lines.append('')
    any_errors = False
    for ev in evaluations:
        errors = []
        for zone, match in (('player', ev['player_match']), ('dealer', ev['dealer_match'])):
            for exp_card, det_card in match['confusions']:
                errors.append(f'{zone}: expected {exp_card}, detected {det_card}')
            for card in match['missed']:
                errors.append(f'{zone}: missed {card}')
            for card in match['spurious']:
                errors.append(f'{zone}: spurious detection {card}')
        if errors:
            any_errors = True
            lines.append(f"- {ev['video']}: " + '; '.join(errors))
    if not any_errors:
        lines.append('No detection errors.')
    lines.append('')

    card = metrics['card_level']
    video = metrics['video_level']

    lines.append('## Metrics')
    lines.append('')
    lines.append('### Card-level (exact card class, player + dealer hands)')
    lines.append('')
    lines.append(f"- True positives: {card['true_positives']}")
    lines.append(f"- False positives: {card['false_positives']}")
    lines.append(f"- False negatives: {card['false_negatives']}")
    lines.append(f"- Precision: {card['precision']}")
    lines.append(f"- Recall: {card['recall']}")
    lines.append(f"- F1 score: {card['f1_score']}")
    lines.append('')
    lines.append('### Video-level')
    lines.append('')
    lines.append(f"- Videos evaluated: {video['videos_evaluated']}")
    lines.append(f"- Videos with all cards correct: {video['all_cards_correct']} "
                 f"({video['card_accuracy'] * 100:.0f}%)")
    lines.append(f"- Videos with correct outcome: {video['outcome_correct']} "
                 f"({video['outcome_accuracy'] * 100:.0f}%)")
    lines.append('')
    lines.append('### Confusion matrix')
    lines.append('')
    lines.append('Rank-level confusion matrix saved to: output/confusion_matrix.png')
    lines.append('')

    matrix = build_confusion_matrix(evaluations)
    labels = RANKS + ['missed']
    header = '| expected \\ detected | ' + ' | '.join(labels) + ' |'
    separator = '|---' * (len(labels) + 1) + '|'
    lines.append(header)
    lines.append(separator)
    for i, rank in enumerate(RANKS):
        if matrix[i].sum() == 0:
            continue
        row = f'| {rank} | ' + ' | '.join(str(v) if v else '' for v in matrix[i]) + ' |'
        lines.append(row)
    lines.append('')

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))


def run_evaluation(results: list):
    ground_truth = load_ground_truth()

    evaluations = []
    skipped = []

    for result in results:
        video_key = Path(result['video']).stem
        if video_key in ground_truth:
            evaluations.append(evaluate_video(result, ground_truth[video_key]))
        else:
            skipped.append(result['video'])

    if not evaluations:
        print('No videos with ground truth found, evaluation skipped.')
        return

    metrics = compute_metrics(evaluations)

    matrix = build_confusion_matrix(evaluations)
    save_confusion_matrix_plot(matrix, OUTPUT_DIR / 'confusion_matrix.png')

    evaluation_output = {
        'generated': datetime.now().isoformat(),
        'metrics': metrics,
        'per_video': [
            {
                'video': ev['video'],
                'expected_player': ev['expected_player'],
                'detected_player': ev['detected_player'],
                'expected_dealer': ev['expected_dealer'],
                'detected_dealer': ev['detected_dealer'],
                'expected_outcome': ev['expected_outcome'],
                'detected_outcome': ev['detected_outcome'],
                'all_cards_correct': ev['all_cards_correct'],
                'outcome_correct': ev['outcome_correct']
            }
            for ev in evaluations
        ],
        'skipped_no_ground_truth': skipped
    }
    with open(OUTPUT_DIR / 'evaluation.json', 'w') as f:
        json.dump(evaluation_output, f, indent=2)

    write_report(evaluations, metrics, skipped, OUTPUT_DIR / 'report.md')

    card = metrics['card_level']
    video = metrics['video_level']
    print(f"Videos evaluated: {video['videos_evaluated']}")
    print(f"Outcome accuracy: {video['outcome_correct']}/{video['videos_evaluated']} "
          f"({video['outcome_accuracy'] * 100:.0f}%)")
    print(f"All cards correct: {video['all_cards_correct']}/{video['videos_evaluated']} "
          f"({video['card_accuracy'] * 100:.0f}%)")
    print(f"Card precision: {card['precision']}, recall: {card['recall']}, "
          f"F1: {card['f1_score']}")
    if skipped:
        print(f"Skipped (no ground truth): {', '.join(skipped)}")
    print('Report saved to: output/report.md')
    print('Confusion matrix saved to: output/confusion_matrix.png')
    print('Evaluation data saved to: output/evaluation.json')


def main():
    results_file = OUTPUT_DIR / 'pipeline_results.json'
    if not results_file.exists():
        print(f'{results_file} not found. Run pipeline.py first.')
        return
    with open(results_file) as f:
        results = json.load(f)['results']
    run_evaluation(results)


if __name__ == '__main__':
    main()
