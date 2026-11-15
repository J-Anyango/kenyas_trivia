"""
Server-authoritative game logic. The client (Android app) only ever
sends intent ("mark word 3 done", "reveal alias for word 7", "end this
turn") — every rule about timing, scoring, aliases, bonuses, rounds
and match-ending is enforced here, not trusted from the client.

This module deliberately contains no Django REST Framework code, so
it can be unit-tested directly and reused if the API layer changes.
"""
from __future__ import annotations

from django.conf import settings
from django.db import transaction
from django.utils import timezone

from .models import (
    ALIAS_LIMIT_PER_TURN,
    ALIAS_POINTS,
    BASE_POINTS,
    EARLY_FINISH_BONUS,
    MAX_TIEBREAK_ROUNDS,
    Match,
    Round,
    Turn,
    TurnWord,
    WORDS_PER_CARD,
)


class GameError(Exception):
    """Raised for any rule violation. Views translate this to a 400."""


def get_card_generator():
    """
    Import lazily so tests / management commands can override
    settings.CARD_GENERATOR without a circular import at module load.
    """
    from django.utils.module_loading import import_string

    generator_path = settings.CARD_GENERATOR
    generator_class = import_string(generator_path)
    return generator_class()


# ---------------------------------------------------------------------
# Starting a match
# ---------------------------------------------------------------------

@transaction.atomic
def start_match(*, team_a_name: str, team_b_name: str, user=None) -> Match:
    match = Match.objects.create(
        team_a_name=team_a_name or "Team A",
        team_b_name=team_b_name or "Team B",
        created_by=user if (user and user.is_authenticated) else None,
        is_guest_match=not (user and user.is_authenticated),
    )
    Round.objects.create(match=match, round_number=1, is_tiebreaker=False)
    return match


# ---------------------------------------------------------------------
# Starting a turn (drawing a card)
# ---------------------------------------------------------------------

def _next_side(round_obj: Round) -> str:
    """Team A goes first in every round; Team B follows."""
    played_sides = set(round_obj.turns.values_list("side", flat=True))
    if "team_a" not in played_sides:
        return "team_a"
    if "team_b" not in played_sides:
        return "team_b"
    raise GameError("Both teams have already played this round.")


@transaction.atomic
def start_turn(*, match: Match) -> Turn:
    if match.status == Match.Status.FINISHED:
        raise GameError("This match has already finished.")

    round_obj = match.rounds.order_by("-round_number").first()
    if round_obj is None:
        raise GameError("Match has no active round.")

    side = _next_side(round_obj)

    turn = Turn.objects.create(round=round_obj, side=side, started_at=timezone.now())

    # No word may repeat anywhere within this lobby (this match), for
    # the match's whole duration — not just within one turn's card.
    # Collect every word already drawn in any turn of this match so
    # far and pass it to the generator to avoid.
    exclude_words = list(
        TurnWord.objects.filter(turn__round__match=match)
        .values_list("word", flat=True)
        .distinct()
    )

    generator = get_card_generator()
    card = generator.generate_card(exclude_words=exclude_words)
    if len(card) != WORDS_PER_CARD:
        raise GameError(
            f"Card generator returned {len(card)} words, expected {WORDS_PER_CARD}."
        )

    TurnWord.objects.bulk_create(
        [
            TurnWord(
                turn=turn,
                order=i,
                word=w["word"],
                alias=w["alias"],
                hint=w["hint"],
            )
            for i, w in enumerate(card)
        ]
    )
    return turn


# ---------------------------------------------------------------------
# During a turn: marking words done, revealing aliases
# ---------------------------------------------------------------------

def _require_turn_open(turn: Turn) -> None:
    if turn.is_complete:
        raise GameError("This turn has already ended.")


def mark_word_done(*, turn: Turn, order: int) -> TurnWord:
    """
    Describer's team guessed this word. Points are computed now
    (10 or 5 depending on whether its alias was revealed) but the
    +50 early-finish bonus is only ever awarded when the turn ends
    (see end_turn), since it depends on the whole card's state.
    """
    _require_turn_open(turn)
    if turn.time_is_up():
        raise GameError("Time is up; this word can no longer be marked.")

    try:
        turn_word = turn.words.get(order=order)
    except TurnWord.DoesNotExist as exc:
        raise GameError("No such word on this card.") from exc

    if turn_word.done:
        return turn_word  # idempotent: already marked, no double-scoring

    turn_word.done = True
    turn_word.points = ALIAS_POINTS if turn_word.alias_used else BASE_POINTS
    turn_word.save(update_fields=["done", "points"])
    return turn_word


def reveal_alias(*, turn: Turn, order: int) -> TurnWord:
    """
    Describer taps the alias button for one word. Costs one of the
    turn's 3 alias reveals, permanently halves that word's points
    (even if the word is later left undone, in which case it scores
    0 anyway), and permanently disqualifies the +50 bonus for this
    turn.
    """
    _require_turn_open(turn)
    if turn.time_is_up():
        raise GameError("Time is up; the alias can no longer be revealed.")

    try:
        turn_word = turn.words.get(order=order)
    except TurnWord.DoesNotExist as exc:
        raise GameError("No such word on this card.") from exc

    if turn_word.alias_used:
        return turn_word  # idempotent

    if turn.alias_reveals_used >= ALIAS_LIMIT_PER_TURN:
        raise GameError(f"Alias already used {ALIAS_LIMIT_PER_TURN} times this turn.")

    turn_word.alias_used = True
    # If the word was already marked done before the alias was
    # revealed, re-price it now. (Normal play reveals the alias
    # first, but this keeps scoring correct either way.)
    if turn_word.done:
        turn_word.points = ALIAS_POINTS
    turn_word.save(update_fields=["alias_used", "points"])

    turn.alias_reveals_used += 1
    turn.save(update_fields=["alias_reveals_used"])
    return turn_word


# ---------------------------------------------------------------------
# Ending a turn: flip, bonus, scoring, round/match progression
# ---------------------------------------------------------------------

@transaction.atomic
def end_turn(*, turn: Turn) -> Turn:
    """
    Called when the 55-second timer elapses, or when the client
    reports all 10 words done early. Either way this is idempotent
    and re-checks everything server-side:
      - words left undone score 0 and get their flip hint shown
        (the hint is already stored on each TurnWord; the API layer
        decides which ones to reveal in the response)
      - +50 bonus iff all 10 words are done AND no alias was used at
        all this turn
      - updates the match's running score for this side
      - if both teams have now played this round, decides whether the
        match continues (next round / tiebreaker) or ends (winner or
        draw)
    """
    if turn.is_complete:
        return turn  # already finalised, nothing to redo

    words = list(turn.words.all())
    all_done = all(w.done for w in words)
    any_alias_used = any(w.alias_used for w in words)

    bonus = EARLY_FINISH_BONUS if (all_done and not any_alias_used) else 0

    turn.score = sum(w.points for w in words) + bonus
    turn.bonus_awarded = bonus > 0
    turn.ended_at = turn.ended_at or timezone.now()
    turn.is_complete = True
    turn.save(update_fields=["score", "bonus_awarded", "ended_at", "is_complete"])

    match = turn.round.match
    match.add_score(turn.side, turn.score)
    match.save(update_fields=["team_a_score", "team_b_score"])

    _advance_match_if_round_complete(match=match, round_obj=turn.round)

    return turn


def _advance_match_if_round_complete(*, match: Match, round_obj: Round) -> None:
    sides_played = set(round_obj.turns.filter(is_complete=True).values_list("side", flat=True))
    if sides_played != {"team_a", "team_b"}:
        return  # round still has a turn to go

    if round_obj.is_tiebreaker:
        _finish_match(match)
        return

    if round_obj.round_number >= 2:
        if match.team_a_score != match.team_b_score:
            _finish_match(match)
            return
        # Level after round 2 (or a later regular round, defensively):
        # exactly one tiebreaker round, then finish no matter what.
        tiebreakers_played = match.rounds.filter(is_tiebreaker=True).count()
        if tiebreakers_played >= MAX_TIEBREAK_ROUNDS:
            _finish_match(match)  # will resolve as a draw
            return
        Round.objects.create(
            match=match,
            round_number=round_obj.round_number + 1,
            is_tiebreaker=True,
        )
        return

    # Regular round finished, scores don't matter yet — start next round.
    Round.objects.create(
        match=match,
        round_number=round_obj.round_number + 1,
        is_tiebreaker=False,
    )


def _finish_match(match: Match) -> None:
    if match.team_a_score > match.team_b_score:
        match.winner = "team_a"
    elif match.team_b_score > match.team_a_score:
        match.winner = "team_b"
    else:
        match.winner = "draw"
    match.status = Match.Status.FINISHED
    match.finished_at = timezone.now()
    match.save(update_fields=["winner", "status", "finished_at"])
