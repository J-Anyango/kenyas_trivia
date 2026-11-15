"""
URL routes for the game API.

NOTE on accounts: email/Google sign-in is not wired up in this
scaffold. When you add it (e.g. django-allauth for Google, or a
simple email+password flow via DRF's token/session auth), the pieces
here that already assume request.user exists — create_match,
my_match_history — will start working for signed-in users without
changes to services.py or models.py.
"""
from django.urls import path

from . import views

urlpatterns = [
    path("matches/", views.create_match, name="create-match"),
    path("matches/<int:match_id>/", views.get_match, name="get-match"),
    path("matches/<int:match_id>/start-turn/", views.start_turn, name="start-turn"),
    path("turns/<int:turn_id>/mark-done/", views.mark_word_done, name="mark-word-done"),
    path("turns/<int:turn_id>/reveal-alias/", views.reveal_alias, name="reveal-alias"),
    path("turns/<int:turn_id>/end/", views.end_turn, name="end-turn"),
    path("me/matches/", views.my_match_history, name="my-match-history"),
]
