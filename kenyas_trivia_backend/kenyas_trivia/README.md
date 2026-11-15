# Kenya's Trivia — game backend scaffold

This is the Django + DRF backend for the Kenya@50-inspired NLP party
game. It implements every game rule you specified, with the model
generator swapped out for a stand-in so the whole game can be built,
played and tested before the trained model exists.

## What's built

- **`game/models.py`** — `Match`, `Round`, `Turn`, `TurnWord`. No word
  database: every card's words are generated live and recorded only
  as the frozen state of a played turn.
- **`game/generator.py`** — `CardGenerator` interface. `StubCardGenerator`
  returns hand-written sample cards (Kenyan figures, places, culture)
  so you can play immediately. Swap it for your trained model later by
  changing **one line** in `config/settings.py` (`CARD_GENERATOR`) —
  nothing else in the codebase changes.
- **`game/services.py`** — all game rules, server-authoritative:
  10 words/card, 55s turn, no skip, alias (max 3, halves points,
  kills the +50 bonus), flip hints, two rounds + one tiebreaker + draw
  on a second tie, round/match progression.
- **`game/views.py` + `game/urls.py`** — the API the Android app calls.
- **`game/admin.py`** — inspect matches/rounds/turns/words in
  `/admin/` while testing.
- **`smoke_test.py`** — plays out full matches through the service
  layer and asserts the scoring/round/tiebreaker rules are all
  correct. All 6 checks currently pass.

## Not built yet (on purpose)

- **Email/Google sign-in.** `Match.created_by` and the
  `/api/me/matches/` endpoint already assume a signed-in
  `request.user`, so wiring up real auth (e.g. `django-allauth` for
  Google) won't require changes to `services.py` or `models.py`.
- **The trained model itself**, and the server that will host it.
  `game/generator.py` has a `TODO` block showing exactly the shape
  `ModelCardGenerator` needs to have.
- **Android app.** This backend is framework-agnostic; nothing here
  assumes what the client is built with.

## Running it

This project uses PostgreSQL (not SQLite). You need Postgres installed
and running first.

1. **Create the database** (one-time, using your own Postgres install):
   ```bash
   createdb kenyas_trivia
   ```
   If that command isn't found or asks for a password you don't know,
   use whatever tool you normally use to administer your local
   Postgres (psql, pgAdmin, etc.) to create a database named
   `kenyas_trivia`.

2. **Create your `.env` file** from the example, and fill in your real
   Postgres username/password:
   ```bash
   cp .env.example .env
   ```
   Then edit `.env` — at minimum set `DB_USER` and `DB_PASSWORD` to
   match your local Postgres, and set `DJANGO_SECRET_KEY` to a real
   random value (the file tells you how to generate one). **Never
   commit `.env` to git** — it's already in `.gitignore`.

3. **Install dependencies, migrate, run:**
   ```bash
   pip install -r requirements.txt
   python manage.py migrate
   python manage.py createsuperuser   # optional, for /admin/
   python manage.py runserver
   ```

If `psycopg2-binary` fails to install, see the troubleshooting note at
the bottom of this file — it's usually a missing system package, not a
problem with this project.

Run the smoke test:
```bash
python smoke_test.py
```

## API summary

| Method | Path                              | Purpose                                   |
|--------|------------------------------------|--------------------------------------------|
| POST   | `/api/matches/`                    | Start a match (guest or signed-in)         |
| GET    | `/api/matches/<id>/`               | Full match state                           |
| POST   | `/api/matches/<id>/start-turn/`    | Draw the next card for whichever team's up |
| POST   | `/api/turns/<id>/mark-done/`       | `{"order": 0-9}` — mark a word guessed     |
| POST   | `/api/turns/<id>/reveal-alias/`    | `{"order": 0-9}` — reveal that word's alias |
| POST   | `/api/turns/<id>/end/`             | End the turn (timer up, or early finish)   |
| GET    | `/api/me/matches/`                 | Signed-in user's match history             |

`start-turn` and `mark-done`/`reveal-alias` return the full card
(word + alias + hint) for every word up front — the **client** is
responsible for not showing the alias/hint until the player asks for
it, but the **server** enforces the 3-alias limit and the 55-second
timer regardless of what the client displays, since the client can't
be trusted.

## No-repeat rule (within one lobby)

No word may repeat anywhere within one lobby (one `Match`), for the
match's whole duration — not just within a single turn's card. This
is enforced in `services.start_turn`, which collects every word
already drawn in any turn of the match so far and passes it to the
generator as `exclude_words`:

```python
CardGenerator.generate_card(self, *, exclude_words: list[str] | None = None)
```

The real, model-backed generator (see the `TODO` in `generator.py`)
sends this list to the hosted model as part of the request body:

```
POST /generate_card
Request:  {"exclude_words": ["Jomo Kenyatta", "Tom Mboya", ...]}
Response: {"words": [ {word, alias, hint} x 10 ]}
```

The model service is responsible for checking its generated words
against `exclude_words` and regenerating any collision — this is a
runtime check, not something the model needs to be trained to do.
There's no time/round limit on this; a new match starts with an empty
exclusion list.

`StubCardGenerator` honours this too, with one documented limitation:
its sample pool is only 15 words, so once a long match has used most
of them it falls back to allowing repeats rather than crashing. The
real model's much larger vocabulary should make this fallback a
non-issue in practice.

## Security notes (read before deploying anywhere)

This scaffold is safe for local development but has a few things that
MUST change before it's ever reachable by anyone other than you:

- **`DJANGO_SECRET_KEY`** — set a real random value in `.env`. Django
  uses this to sign sessions and tokens; a known/default key lets
  someone forge them.
- **`DJANGO_DEBUG=False`** — once deployed. `DEBUG=True` shows full
  error pages (code, settings, query details) to anyone who triggers
  a bug. Fine on your own laptop, a serious leak anywhere else.
- **`DJANGO_ALLOWED_HOSTS`** — set to your real server's domain/IP.
- **Database password** — never hardcode it; always via `.env`
  (already wired up this way).
- **API permissions** — every gameplay endpoint currently uses
  `AllowAny` (anyone can create matches, mark words done, etc. without
  logging in) because guest play is a stated requirement. Only
  `/api/me/matches/` requires login. This is intentional for now, but
  means nothing stops someone from hammering the API with fake
  matches — revisit this (e.g. rate limiting) before a public launch.
- **`/admin/`** — protected by Django's normal login, but only as
  strong as your superuser password. Use a real one, not something
  guessable.

## Troubleshooting

- **`psycopg2-binary` fails to install:** this usually means a system
  package is missing (e.g. `libpq-dev` on Ubuntu/Debian, or Xcode
  command line tools on macOS). Install that, then retry
  `pip install -r requirements.txt`.
- **`django.db.utils.OperationalError: connection refused`:** Postgres
  isn't running, or `DB_HOST`/`DB_PORT` in `.env` don't match your
  install. Confirm Postgres is running and check those values.
- **`FATAL: password authentication failed`:** `DB_USER`/`DB_PASSWORD`
  in `.env` don't match your Postgres setup.
