from rest_framework import serializers

from .models import ALIAS_LIMIT_PER_TURN, TURN_SECONDS, Match, Round, Turn, TurnWord


class TurnWordSerializer(serializers.ModelSerializer):
    """
    Full word state, used for the describer's own view mid-turn.
    `hint` is included here because only the describer's device shows
    it, and only after the turn ends / the client requests the flip —
    the view layer decides when to include hints, not this
    serializer's shape.
    """

    class Meta:
        model = TurnWord
        fields = ["order", "word", "alias", "hint", "done", "alias_used", "points"]


class TurnWordPublicSerializer(serializers.ModelSerializer):
    """
    What guessers would see if the app ever mirrors turn state to a
    second screen: the word itself is intentionally omitted before
    the flip. Not wired into any endpoint yet — kept here as the
    obvious extension point.
    """

    class Meta:
        model = TurnWord
        fields = ["order", "done"]


class TurnSerializer(serializers.ModelSerializer):
    words = TurnWordSerializer(many=True, read_only=True)
    seconds_remaining = serializers.SerializerMethodField()

    class Meta:
        model = Turn
        fields = [
            "id",
            "round",
            "side",
            "started_at",
            "ended_at",
            "alias_reveals_used",
            "alias_reveals_remaining",
            "score",
            "bonus_awarded",
            "is_complete",
            "seconds_remaining",
            "words",
        ]

    alias_reveals_remaining = serializers.SerializerMethodField()

    def get_alias_reveals_remaining(self, obj: Turn) -> int:
        return max(0, ALIAS_LIMIT_PER_TURN - obj.alias_reveals_used)

    def get_seconds_remaining(self, obj: Turn) -> float:
        return max(0.0, TURN_SECONDS - obj.seconds_elapsed())


class RoundSerializer(serializers.ModelSerializer):
    turns = TurnSerializer(many=True, read_only=True)

    class Meta:
        model = Round
        fields = ["id", "round_number", "is_tiebreaker", "turns"]


class MatchSerializer(serializers.ModelSerializer):
    rounds = RoundSerializer(many=True, read_only=True)

    class Meta:
        model = Match
        fields = [
            "id",
            "team_a_name",
            "team_b_name",
            "status",
            "team_a_score",
            "team_b_score",
            "winner",
            "is_guest_match",
            "created_at",
            "finished_at",
            "rounds",
        ]


class MatchSummarySerializer(serializers.ModelSerializer):
    """Lightweight shape for list views / match history."""

    class Meta:
        model = Match
        fields = [
            "id",
            "team_a_name",
            "team_b_name",
            "status",
            "team_a_score",
            "team_b_score",
            "winner",
            "created_at",
            "finished_at",
        ]
