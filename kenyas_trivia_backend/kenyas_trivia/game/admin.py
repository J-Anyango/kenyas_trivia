from django.contrib import admin

from .models import Match, Round, Turn, TurnWord


class TurnWordInline(admin.TabularInline):
    model = TurnWord
    extra = 0
    readonly_fields = ["order", "word", "alias", "hint", "done", "alias_used", "points"]
    can_delete = False


class TurnInline(admin.TabularInline):
    model = Turn
    extra = 0
    show_change_link = True
    fields = ["side", "score", "bonus_awarded", "is_complete", "started_at"]
    readonly_fields = fields


@admin.register(Turn)
class TurnAdmin(admin.ModelAdmin):
    list_display = ["id", "round", "side", "score", "bonus_awarded", "is_complete"]
    inlines = [TurnWordInline]


class RoundInline(admin.TabularInline):
    model = Round
    extra = 0
    show_change_link = True
    fields = ["round_number", "is_tiebreaker"]
    readonly_fields = fields


@admin.register(Round)
class RoundAdmin(admin.ModelAdmin):
    list_display = ["id", "match", "round_number", "is_tiebreaker"]
    inlines = [TurnInline]


@admin.register(Match)
class MatchAdmin(admin.ModelAdmin):
    list_display = [
        "id",
        "team_a_name",
        "team_b_name",
        "team_a_score",
        "team_b_score",
        "status",
        "winner",
        "is_guest_match",
        "created_at",
    ]
    inlines = [RoundInline]
