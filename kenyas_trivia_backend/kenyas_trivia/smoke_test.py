"""
Manual smoke test: plays out a full match through services.py and
asserts the agreed rules hold. Run with:
    python manage.py shell < smoke_test.py
(or DJANGO_SETTINGS_MODULE set + python smoke_test.py after django.setup())
"""
import django
import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

from django.utils import timezone
from game import services
from game.models import BASE_POINTS, ALIAS_POINTS, EARLY_FINISH_BONUS, WORDS_PER_CARD

def play_turn(match, *, mark_all=True, use_aliases=0, force_time_up=False):
    turn = services.start_turn(match=match)
    words = list(turn.words.all())

    for i in range(use_aliases):
        services.reveal_alias(turn=turn, order=words[i].order)

    if mark_all:
        for w in words:
            services.mark_word_done(turn=turn, order=w.order)

    if force_time_up:
        turn.started_at = timezone.now() - timezone.timedelta(seconds=60)
        turn.save(update_fields=["started_at"])

    services.end_turn(turn=turn)
    turn.refresh_from_db()
    return turn


# --- Test 1: perfect turn, no alias -> full 150 (100 + 50 bonus) ---
m1 = services.start_match(team_a_name="Alpha", team_b_name="Beta")
t1 = play_turn(m1, mark_all=True, use_aliases=0)
assert t1.score == WORDS_PER_CARD * BASE_POINTS + EARLY_FINISH_BONUS, t1.score
assert t1.bonus_awarded is True
print("Test 1 passed: perfect no-alias turn scores", t1.score)

# --- Test 2: perfect turn WITH alias -> no bonus, mixed points ---
m2 = services.start_match(team_a_name="Alpha", team_b_name="Beta")
t2 = play_turn(m2, mark_all=True, use_aliases=3)
expected = 3 * ALIAS_POINTS + 7 * BASE_POINTS
assert t2.score == expected, (t2.score, expected)
assert t2.bonus_awarded is False
print("Test 2 passed: alias used -> bonus lost, score", t2.score, "== expected", expected)

# --- Test 3: alias limited to 3 ---
m3 = services.start_match(team_a_name="Alpha", team_b_name="Beta")
t3 = services.start_turn(match=m3)
words3 = list(t3.words.all())
for w in words3[:3]:
    services.reveal_alias(turn=t3, order=w.order)
try:
    services.reveal_alias(turn=t3, order=words3[3].order)
    raise SystemExit("Test 3 FAILED: 4th alias reveal should have been rejected")
except services.GameError:
    print("Test 3 passed: 4th alias reveal correctly rejected")
services.end_turn(turn=t3)

# --- Test 4: partial turn (some words missed) scores only completed ones ---
m4 = services.start_match(team_a_name="Alpha", team_b_name="Beta")
t4 = services.start_turn(match=m4)
words4 = list(t4.words.all())
for w in words4[:6]:
    services.mark_word_done(turn=t4, order=w.order)
services.end_turn(turn=t4)
t4.refresh_from_db()
assert t4.score == 6 * BASE_POINTS, t4.score
assert t4.bonus_awarded is False
print("Test 4 passed: partial turn (6/10) scores", t4.score, "no bonus")

# --- Test 5: full match progression -> 2 rounds, winner decided, no tiebreaker ---
m5 = services.start_match(team_a_name="Alpha", team_b_name="Beta")
# Round 1: A gets full marks, B gets nothing
play_turn(m5, mark_all=True, use_aliases=0)   # team_a turn 1
t = services.start_turn(match=m5)             # team_b turn 1 (round still 1)
services.end_turn(turn=t)                      # B does nothing -> 0
# Round 2 should now exist automatically
play_turn(m5, mark_all=True, use_aliases=0)   # team_a turn 2
t = services.start_turn(match=m5)             # team_b turn 2
services.end_turn(turn=t)
m5.refresh_from_db()
assert m5.status == m5.Status.FINISHED, m5.status
assert m5.winner == "team_a", m5.winner
assert m5.rounds.count() == 2, m5.rounds.count()  # no tiebreaker needed
print("Test 5 passed: match finished after 2 rounds, winner =", m5.winner)

# --- Test 6: tie after 2 rounds -> exactly one tiebreaker round, then draw ---
m6 = services.start_match(team_a_name="Alpha", team_b_name="Beta")
for _ in range(2):  # rounds 1 and 2, both teams score identically (0 each, all missed)
    t = services.start_turn(match=m6)
    services.end_turn(turn=t)
    t = services.start_turn(match=m6)
    services.end_turn(turn=t)
m6.refresh_from_db()
assert m6.status != m6.Status.FINISHED, "should not finish yet, scores tied"
assert m6.rounds.count() == 3, m6.rounds.count()
assert m6.rounds.get(round_number=3).is_tiebreaker is True
# Play the tiebreaker, still tied -> draw
t = services.start_turn(match=m6)
services.end_turn(turn=t)
t = services.start_turn(match=m6)
services.end_turn(turn=t)
m6.refresh_from_db()
assert m6.status == m6.Status.FINISHED
assert m6.winner == "draw", m6.winner
assert m6.rounds.count() == 3, "must not create a second tiebreaker"
print("Test 6 passed: single tiebreaker then draw, rounds =", m6.rounds.count())

# --- Test 7: exclude_words is correctly built from the match's
# already-played words, and the generator honours it when the word
# pool allows it to. (The stand-in's pool is intentionally tiny —
# 15 words — so this also proves the documented fallback doesn't
# kick in until genuinely necessary.)
from game.generator import StubCardGenerator

gen = StubCardGenerator()
pool_words = [w["word"] for w in gen.SAMPLE_WORDS]
assert len(pool_words) == 15

# a) direct generator check: excluding 5 of 15 still leaves 10 to draw
# from cleanly -> zero overlap expected, no fallback needed.
excl = pool_words[:5]
card = gen.generate_card(exclude_words=excl)
drawn = [w["word"] for w in card]
assert not (set(drawn) & set(excl)), "Test 7a FAILED: generator drew an excluded word unnecessarily"
assert len(drawn) == WORDS_PER_CARD
print("Test 7a passed: generator avoids exclude_words when the pool allows it")

# b) services wiring check: after turn 1, the match's exclude list
# passed to the generator must equal exactly the words drawn in turn 1.
m7 = services.start_match(team_a_name="Alpha", team_b_name="Beta")
t1 = services.start_turn(match=m7)
turn1_words = set(w.word for w in t1.words.all())
services.end_turn(turn=t1)

from unittest.mock import patch
captured = {}
original_generate = StubCardGenerator.generate_card
def spy(self, *, exclude_words=None):
    captured["exclude_words"] = set(exclude_words or [])
    return original_generate(self, exclude_words=exclude_words)
with patch.object(StubCardGenerator, "generate_card", spy):
    t2 = services.start_turn(match=m7)
    services.end_turn(turn=t2)
assert captured["exclude_words"] == turn1_words, (
    "Test 7b FAILED: exclude_words sent to generator did not match turn 1's words",
    captured["exclude_words"], turn1_words,
)
print("Test 7b passed: services.start_turn correctly builds exclude_words from prior turns in the match")

print("\nAll smoke tests passed.")
