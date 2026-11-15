"""
Data model for the Kenya@50-inspired NLP party game.

Design notes (matching the agreed game rules):
- There is no stored word database. Every card's 10 words, aliases and
  hints are produced at play time by a CardGenerator (see generator.py).
  What we persist is the record of a *played* card (TurnWord), not a
  catalog of words.
- A Match has 2 regular rounds, alternating teams, plus at most ONE
  tiebreaker round if scores are level after round 2. If still level
  after the tiebreaker, the match ends in a draw.
- A Turn is one team's 55-second attempt at one 10-word card.
- Scoring (server-authoritative, see services.py):
    * word done, no alias   -> 10 points
    * word done, alias used -> 5 points
    * word not done         -> 0 points
    * all 10 done before time AND no alias used anywhere on the card
      -> +50 bonus (lost entirely if any alias was used this turn)
- Alias: generated for every word up front, but only revealed to the
  describer for words they tap (max 3 reveals per turn). Revealing
  counts as "used" immediately, even if the word is later left undone.
"""
from django.conf import settings
from django.db import models
from django.utils import timezone

TURN_SECONDS = 55
WORDS_PER_CARD = 10
BASE_POINTS = 10
ALIAS_POINTS = 5
ALIAS_LIMIT_PER_TURN = 3
EARLY_FINISH_BONUS = 50
MAX_TIEBREAK_ROUNDS = 1


class Match(models.Model):
    """
    One full game between two teams. Team identity is just a label the
    players choose themselves — the game does not manage team rosters.
    """

    class Status(models.TextChoices):
        IN_PROGRESS = "in_progress", "In progress"
        FINISHED = "finished", "Finished"

    team_a_name = models.CharField(max_length=100, default="Team A")
    team_b_name = models.CharField(max_length=100, default="Team B")

    # Signed-in account that started the match, if any. Guests can play
    # a full match with nothing saved beyond this row's lifetime use.
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="matches",
    )
    is_guest_match = models.BooleanField(default=True)

    status = models.CharField(
        max_length=20, choices=Status.choices, default=Status.IN_PROGRESS
    )

    team_a_score = models.PositiveIntegerField(default=0)
    team_b_score = models.PositiveIntegerField(default=0)

    # "team_a" / "team_b" / "draw" once finished, else blank.
    winner = models.CharField(max_length=10, blank=True, default="")

    created_at = models.DateTimeField(auto_now_add=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    def team_name(self, side: str) -> str:
        return self.team_a_name if side == "team_a" else self.team_b_name

    def add_score(self, side: str, points: int) -> None:
        if side == "team_a":
            self.team_a_score += points
        else:
            self.team_b_score += points

    def __str__(self):
        return f"Match {self.pk}: {self.team_a_name} vs {self.team_b_name}"


class Round(models.Model):
    """
    Round 1 and round 2 are regular rounds. Round 3, if it happens, is
    the single tiebreaker round. Each round has exactly one turn per
    team (created as needed, not pre-created).
    """

    match = models.ForeignKey(Match, on_delete=models.CASCADE, related_name="rounds")
    round_number = models.PositiveSmallIntegerField()
    is_tiebreaker = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("match", "round_number")
        ordering = ["round_number"]

    def __str__(self):
        label = "Tiebreaker" if self.is_tiebreaker else f"Round {self.round_number}"
        return f"{label} of match {self.match_id}"


class Turn(models.Model):
    """
    One team's 55-second attempt at one generated 10-word card.
    The server owns timing: `started_at` is set when the card is
    drawn, and any word-marking or alias-reveal request arriving after
    TURN_SECONDS (+ small grace) is rejected by services.py, not by
    the client.
    """

    round = models.ForeignKey(Round, on_delete=models.CASCADE, related_name="turns")
    side = models.CharField(max_length=10, choices=[("team_a", "Team A"), ("team_b", "Team B")])

    started_at = models.DateTimeField(default=timezone.now)
    ended_at = models.DateTimeField(null=True, blank=True)

    # Denormalised counters, kept in sync by services.py so the API
    # doesn't have to recompute them from TurnWord on every read.
    alias_reveals_used = models.PositiveSmallIntegerField(default=0)
    score = models.PositiveIntegerField(default=0)
    bonus_awarded = models.BooleanField(default=False)
    is_complete = models.BooleanField(default=False)

    class Meta:
        ordering = ["id"]

    def seconds_elapsed(self) -> float:
        end = self.ended_at or timezone.now()
        return (end - self.started_at).total_seconds()

    def time_is_up(self) -> bool:
        return self.seconds_elapsed() >= TURN_SECONDS

    def __str__(self):
        return f"Turn {self.pk} ({self.side}) in round {self.round_id}"


class TurnWord(models.Model):
    """
    One of the 10 words generated for a Turn's card, plus the
    play-time state of that word. `word`, `alias` and `hint` are a
    frozen copy of what the generator produced for this turn — never a
    foreign key into a word catalog, because there isn't one.
    """

    turn = models.ForeignKey(Turn, on_delete=models.CASCADE, related_name="words")
    order = models.PositiveSmallIntegerField()  # 0..9, position on the card

    word = models.CharField(max_length=200)
    alias = models.CharField(max_length=100)
    hint = models.TextField()

    done = models.BooleanField(default=False)
    alias_used = models.BooleanField(default=False)
    points = models.PositiveSmallIntegerField(default=0)

    class Meta:
        ordering = ["order"]
        unique_together = ("turn", "order")

    def __str__(self):
        return f"{self.word} (turn {self.turn_id}, #{self.order})"
