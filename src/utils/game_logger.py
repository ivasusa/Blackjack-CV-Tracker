#!/usr/bin/env python3

import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from datetime import datetime


@dataclass
class CardEvent:
    sequence: int
    card: str
    zone: str
    frame: int
    timestamp_seconds: float
    event_type: str
    visible: bool


@dataclass
class GameEvent:
    frame: int
    timestamp_seconds: float
    event_type: str
    details: Dict


class GameLogger:

    def __init__(self, video_name: str, fps: float = 30.0):
        self.video_name = video_name
        self.fps = fps
        self.card_events: List[CardEvent] = []
        self.game_events: List[GameEvent] = []
        self.player_cards: List[str] = []
        self.dealer_visible_cards: List[str] = []
        self.dealer_hidden_card: Optional[str] = None
        self.outcome: Optional[str] = None
        self.timestamp_created = datetime.now().isoformat()

    def add_card_event(self, card: str, zone: str, frame: int,
                       event_type: str = 'card_placed', visible: bool = True):
        event = CardEvent(
            sequence=len(self.card_events) + 1,
            card=card,
            zone=zone,
            frame=frame,
            timestamp_seconds=frame / self.fps,
            event_type=event_type,
            visible=visible
        )
        self.card_events.append(event)

    def add_game_event(self, frame: int, event_type: str, details: Dict):
        event = GameEvent(
            frame=frame,
            timestamp_seconds=frame / self.fps,
            event_type=event_type,
            details=details
        )
        self.game_events.append(event)

    def set_final_state(self, player_cards: List[str],
                        dealer_visible_cards: List[str],
                        dealer_hidden_card: Optional[str],
                        outcome: str):
        self.player_cards = player_cards
        self.dealer_visible_cards = dealer_visible_cards
        self.dealer_hidden_card = dealer_hidden_card
        self.outcome = outcome

    def to_dict(self) -> Dict:
        return {
            'metadata': {
                'video': self.video_name,
                'fps': self.fps,
                'timestamp_created': self.timestamp_created,
                'total_card_events': len(self.card_events),
                'total_game_events': len(self.game_events)
            },
            'final_state': {
                'player_cards': self.player_cards,
                'dealer_visible_cards': self.dealer_visible_cards,
                'dealer_hidden_card': self.dealer_hidden_card,
                'outcome': self.outcome
            },
            'card_timeline': [asdict(e) for e in self.card_events],
            'game_timeline': [asdict(e) for e in self.game_events]
        }

    def save_json(self, output_path: Path):
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f'Game log saved: {output_path}')
