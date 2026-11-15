"""
CardGenerator: the one seam between the game engine and "wherever the
words come from". The game only ever talks to this interface, so
swapping the stand-in generator for a real call to your trained model
(Option 1 — hosted on your own server) is a one-line change in
settings.py and touches nothing else in models.py, services.py or
views.py.

CARD FORMAT (the contract the model must satisfy):
    A card is a list of exactly WORDS_PER_CARD dicts, each:
        {
            "word":  str,   # the target word/name shown to the describer
            "alias": str,   # ONE word, the optional half-points clue
            "hint":  str,   # one short English sentence, shown on the
                             # flip for words the team missed. For
                             # historic figures/events it must describe
                             # their cultural relevance or what they are
                             # popularly known for. May name the word.
        }
    No categories, no era/generation tagging, no forbidden words —
    per the agreed rules. Sheng is out of scope entirely.

NO-REPEAT RULE:
    Within one lobby (one Match), no word may repeat across the whole
    match — not just within a single turn's card. generate_card()
    therefore takes the list of words already drawn anywhere in this
    match so far, and must return 10 words that avoid all of them.
    There is no time/round limit on this — it's scoped to the lobby's
    entire lifetime, and resets to empty for a new match.

    The real ModelCardGenerator (see TODO below) sends this list to
    the hosted model as `exclude_words` in the request body:

        POST /generate_card
        Request:  {"exclude_words": ["Jomo Kenyatta", "Tom Mboya", ...]}
        Response: {"words": [ {word, alias, hint} x 10 ]}

    The model service is responsible for generating words, checking
    each against exclude_words, and regenerating any collision before
    returning — this is a runtime check, not something the model is
    trained to do.
"""
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import TypedDict

from .models import WORDS_PER_CARD


class CardWord(TypedDict):
    word: str
    alias: str
    hint: str


class CardGenerator(ABC):
    @abstractmethod
    def generate_card(self, *, exclude_words: list[str] | None = None) -> list[CardWord]:
        """
        Return exactly WORDS_PER_CARD CardWord dicts for one turn,
        none of whose `word` value appears in `exclude_words` (every
        word already drawn anywhere in this match so far).
        """
        raise NotImplementedError


class StubCardGenerator(CardGenerator):
    """
    Hand-written sample cards so the game engine, scoring, and API can
    be built and tested before the trained model exists. Replace with
    ModelCardGenerator (or similar) once the model is trained and
    hosted — see the TODO at the bottom of this file.

    Every entry here follows the same rule the real model must follow:
    the hint describes cultural relevance or what the word is
    popularly known for, in one short English sentence.
    """

    SAMPLE_WORDS: list[CardWord] = [
        {"word": "Jomo Kenyatta", "alias": "President",
         "hint": "Kenya's first president, remembered as the founding father of the nation."},
        {"word": "Dedan Kimathi", "alias": "Freedom fighter",
         "hint": "A Mau Mau leader honoured today with statues and a university named after him."},
        {"word": "Mekatilili wa Menza", "alias": "Resistance",
         "hint": "A Giriama elder celebrated for leading resistance against colonial rule."},
        {"word": "Tom Mboya", "alias": "Airlift",
         "hint": "A independence-era leader remembered for the student airlifts and a Nairobi street bearing his name."},
        {"word": "Maasai Mara", "alias": "Safari",
         "hint": "A famous game reserve known worldwide for the wildebeest migration."},
        {"word": "Madaraka Day", "alias": "Holiday",
         "hint": "A national holiday marking the day Kenya achieved internal self-rule."},
        {"word": "Nyayo", "alias": "Footsteps",
         "hint": "A word meaning footsteps, associated with President Moi's era and philosophy."},
        {"word": "Ugali", "alias": "Staple",
         "hint": "A maize-meal dish that is a staple food eaten across Kenyan households."},
        {"word": "Mount Kenya", "alias": "Peak",
         "hint": "The country's tallest mountain and the namesake of the nation."},
        {"word": "Wangari Maathai", "alias": "Trees",
         "hint": "A Nobel Peace Prize laureate known for founding the Green Belt Movement."},
        {"word": "Harambee", "alias": "Together",
         "hint": "A Swahili word for communal self-help, and Kenya's national motto."},
        {"word": "Lupita Nyong'o", "alias": "Actress",
         "hint": "An Oscar-winning Kenyan actress celebrated internationally."},
        {"word": "Eliud Kipchoge", "alias": "Marathon",
         "hint": "A long-distance runner famous for being the first to run a marathon in under two hours."},
        {"word": "Nairobi", "alias": "Capital",
         "hint": "Kenya's capital city, often nicknamed the Green City in the Sun."},
        {"word": "Mashujaa Day", "alias": "Heroes",
         "hint": "A national holiday honouring Kenya's heroes and heroines."},
    ]

    def generate_card(self, *, exclude_words: list[str] | None = None) -> list[CardWord]:
        exclude = set(exclude_words or [])
        available = [w for w in self.SAMPLE_WORDS if w["word"] not in exclude]
        if len(available) < WORDS_PER_CARD:
            # The stand-in's sample pool is small (15 words) and will
            # run out well before a real model would. Rather than
            # crash mid-match, fall back to allowing repeats once the
            # pool is exhausted — this is a stand-in limitation only;
            # the real model has a far larger vocabulary and should
            # not need this fallback in practice.
            available = self.SAMPLE_WORDS
        return random.sample(available, k=WORDS_PER_CARD)


# ---------------------------------------------------------------------
# TODO (later stage): ModelCardGenerator
#
# When the trained model is ready and hosted (Option 1 — your own
# server, called by Django), implement a second class here, e.g.:
#
#   class ModelCardGenerator(CardGenerator):
#       def __init__(self, base_url: str):
#           self.base_url = base_url
#
#       def generate_card(self, *, exclude_words=None) -> list[CardWord]:
#           response = requests.post(
#               f"{self.base_url}/generate_card",
#               json={"exclude_words": exclude_words or []},
#           )
#           response.raise_for_status()
#           data = response.json()
#           # validate it has exactly WORDS_PER_CARD entries with the
#           # word/alias/hint keys, and that none of them are in
#           # exclude_words, before returning it
#           return data["words"]
#
# Then in settings.py, change CARD_GENERATOR to point at it instead of
# StubCardGenerator. Nothing in services.py, models.py or views.py
# needs to change beyond what's already wired up for exclude_words.
# ---------------------------------------------------------------------
