"""
API surface for the Android client.

Every endpoint that changes game state re-derives what it needs from
the database and from services.py rather than trusting the client's
idea of scores or timing — see services.py for why.

Auth: match history/scores are only saved against a signed-in user
(see models.Match.created_by / is_guest_match). Guests can call every
endpoint below; their matches just aren't attributed to an account.
Wiring up email/Google login itself is a separate, later step (not
part of this scaffold) — see the note in urls.py.
"""
from django.shortcuts import get_object_or_404
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response

from . import services
from .models import Match, Turn
from .serializers import MatchSerializer, MatchSummarySerializer, TurnSerializer


def _error(message: str, code=status.HTTP_400_BAD_REQUEST):
    return Response({"detail": message}, status=code)


@api_view(["POST"])
@permission_classes([AllowAny])
def create_match(request):
    """
    POST /api/matches/
    body: {"team_a_name": "...", "team_b_name": "..."}
    Works for guests (request.user is anonymous) and signed-in users.
    """
    team_a_name = request.data.get("team_a_name", "Team A")
    team_b_name = request.data.get("team_b_name", "Team B")
    user = request.user if request.user and request.user.is_authenticated else None

    match = services.start_match(team_a_name=team_a_name, team_b_name=team_b_name, user=user)
    return Response(MatchSerializer(match).data, status=status.HTTP_201_CREATED)


@api_view(["GET"])
@permission_classes([AllowAny])
def get_match(request, match_id: int):
    """GET /api/matches/<id>/ — full state, including all rounds/turns/words."""
    match = get_object_or_404(Match, pk=match_id)
    return Response(MatchSerializer(match).data)


@api_view(["POST"])
@permission_classes([AllowAny])
def start_turn(request, match_id: int):
    """
    POST /api/matches/<id>/start-turn/
    Draws a new card for whichever side is next to play. Returns the
    turn including the full card (word + alias + hint per word) for
    the describer's device. The client is responsible for not showing
    the hint or the alias until the player asks for it — but note the
    server enforces alias-reveal limits and timing regardless of what
    the client displays.
    """
    match = get_object_or_404(Match, pk=match_id)
    try:
        turn = services.start_turn(match=match)
    except services.GameError as exc:
        return _error(str(exc))
    return Response(TurnSerializer(turn).data, status=status.HTTP_201_CREATED)


@api_view(["POST"])
@permission_classes([AllowAny])
def mark_word_done(request, turn_id: int):
    """
    POST /api/turns/<id>/mark-done/
    body: {"order": 0-9}
    """
    turn = get_object_or_404(Turn, pk=turn_id)
    order = request.data.get("order")
    if order is None:
        return _error("`order` is required.")
    try:
        services.mark_word_done(turn=turn, order=int(order))
    except services.GameError as exc:
        return _error(str(exc))
    turn.refresh_from_db()
    return Response(TurnSerializer(turn).data)


@api_view(["POST"])
@permission_classes([AllowAny])
def reveal_alias(request, turn_id: int):
    """
    POST /api/turns/<id>/reveal-alias/
    body: {"order": 0-9}
    """
    turn = get_object_or_404(Turn, pk=turn_id)
    order = request.data.get("order")
    if order is None:
        return _error("`order` is required.")
    try:
        services.reveal_alias(turn=turn, order=int(order))
    except services.GameError as exc:
        return _error(str(exc))
    turn.refresh_from_db()
    return Response(TurnSerializer(turn).data)


@api_view(["POST"])
@permission_classes([AllowAny])
def end_turn(request, turn_id: int):
    """
    POST /api/turns/<id>/end/
    Called by the client when the 55s countdown elapses, or
    immediately if all 10 words were marked done early. Safe to call
    more than once (idempotent) and safe to call "too early" — the
    server does not require time_is_up() to end a turn, since an
    early-finish is a valid, bonus-earning way to end it.
    Returns the full match state so the client can show the flip
    (hints for missed words), updated scores, and — if this was the
    round's last turn — the outcome of round/match progression.
    """
    turn = get_object_or_404(Turn, pk=turn_id)
    services.end_turn(turn=turn)
    match = turn.round.match
    return Response(MatchSerializer(match).data)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def my_match_history(request):
    """
    GET /api/me/matches/
    Saved scores and match history for signed-in accounts only —
    guests have no history to fetch, by design.
    """
    matches = Match.objects.filter(created_by=request.user).order_by("-created_at")
    return Response(MatchSummarySerializer(matches, many=True).data)
