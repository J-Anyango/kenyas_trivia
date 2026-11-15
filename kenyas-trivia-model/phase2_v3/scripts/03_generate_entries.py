#!/usr/bin/env python3
"""
Phase 2, step 1 (runs on Kaggle GPU): turn each Wikipedia article into a
grounded {word, alias, hint} entry with an open-weight LLM (Qwen2.5-7B-Instruct).

Stage A  generate : the LLM reads ONLY the article text and writes four candidate
                    one-word aliases, a hint, and a `hook` flag (does the text show
                    a living cultural hook?)
Stage B  repair   : hints that are too long, use filler words, or use time-relative
                    words ("current president") get ONE rewrite pass
Stage C  ground   : the same LLM acts as a strict checker: is every claim in the
                    hint supported by the article text? (SUPPORTED / UNSUPPORTED)

`word` is NOT written by the LLM. It is the article title minus any trailing
"(...)" (and a trailing "people"), so the target word can never be invented.

Resumable: titles already present in the output file are skipped.
Use --prev to continue from an earlier run's output (copied into --out first).

Kaggle usage (GPU on, Internet on):
    !pip -q install bitsandbytes accelerate
    !python <dir>/03_generate_entries.py --articles <dir>/articles.jsonl --out /kaggle/working/entries_raw.jsonl --limit 20
Full run in chunks (each Kaggle session has a 12 hour limit), e.g. 1200 per session:
    ... --limit 1200
    next session: add the previous notebook's output as input data, then
    ... --limit 1200 --prev /kaggle/input/<previous-output>/entries_raw.jsonl

Local pipeline test without a GPU:
    python scripts/03_generate_entries.py --stub --limit 5
"""
import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

MODEL_DEFAULT = "Qwen/Qwen2.5-7B-Instruct"
MAX_ARTICLE_CHARS = 5000   # first ~1,200 tokens: the intro holds the "known for" facts

SYSTEM_GEN = (
    "You write clues for a Kenyan word-guessing party game. You are given a "
    "Wikipedia article. Use ONLY facts stated in the article text. Never use "
    "outside knowledge. If the text does not say it, do not write it.\n\n"
    "Return ONLY a JSON object with exactly these keys:\n"
    '  "aliases": a list of 4 DIFFERENT clue words, best first. Each is ONE ordinary '
    "dictionary word (letters only) that points toward the subject without being "
    'part of its name, such as a role, type or category (e.g. "President", '
    '"Holiday", "Marathoner", "Peak"). Never join two words into one (not '
    '"GamingCompany"), and never use a word that appears in the title.\n'
    '  "hint": ONE plain sentence of at most 18 words saying why Kenyans know this '
    "subject or what it is popularly known for. If the text links the subject to "
    "Kenya or Kenyans, say so. For people and events, say what they are remembered "
    "or celebrated for, not a dry biographical fact. Do not use filler words such "
    "as renowned, significant, leading or notable. Do not use time-relative words "
    "such as current, currently, latest or recent. The sentence may mention the "
    "title.\n"
    '  "hook": true only if the text shows the subject still matters to Kenyans '
    "today (still referenced, celebrated, named after, or widely recognised); false "
    "if it is only administratively or historically important, or obscure. If "
    'false, set "aliases" to [] and "hint" to "".\n\n'
    "English only. No slang. No quotation marks inside the values."
)

FEWSHOT = [
    (
        "Madaraka Day",
        "Madaraka Day is a public holiday in Kenya observed every year to "
        "commemorate the day Kenya attained internal self-rule in 1963.",
        {"aliases": ["Holiday", "Celebration", "Anniversary", "Independence"],
         "hint": "Madaraka Day is a national holiday Kenyans mark every year to remember the start of internal self-rule in 1963.",
         "hook": True},
    ),
    (
        "Dedan Kimathi",
        "Dedan Kimathi was a Kenyan leader of the Mau Mau uprising against British "
        "colonial rule. A university and streets in Kenya are named after him, and "
        "he is widely remembered as a freedom fighter.",
        {"aliases": ["Rebel", "Fighter", "Guerrilla", "Hero"],
         "hint": "Dedan Kimathi is remembered as a Mau Mau freedom fighter, and a university and streets across Kenya carry his name.",
         "hook": True},
    ),
    (
        "Mount Kenya",
        "Mount Kenya is the highest mountain in Kenya and the second-highest in "
        "Africa. The country of Kenya is named after the mountain.",
        {"aliases": ["Peak", "Summit", "Highland", "Volcano"],
         "hint": "Mount Kenya is the country's highest peak, and Kenya takes its name from it.",
         "hook": True},
    ),
    (
        "Example Ward Office",
        "The Example Ward Office is an administrative building that housed the "
        "ward clerk between 1952 and 1968.",
        {"aliases": [], "hint": "", "hook": False},
    ),
]

SYSTEM_REPAIR = (
    "You edit one-sentence clues for a Kenyan word-guessing game. Use ONLY facts "
    "stated in the article text. Return ONLY the corrected sentence, nothing else."
)

SYSTEM_CHECK = (
    "You are a strict fact checker. You are given an article text and a CLAIM. "
    "Answer SUPPORTED only if every factual statement in the CLAIM is directly "
    "stated in the text. Otherwise answer UNSUPPORTED. Reply with exactly one word."
)

FILLER = re.compile(r"\b(renowned|significant|leading|notable|prominent)\b", re.I)
STALE = re.compile(r"\b(current|currently|latest|newest|recently)\b", re.I)


# ------------------------------------------------------------ prompt/parse ---
def clean_word(title):
    """Article title -> the target word shown in the game."""
    w = re.sub(r"\s*\([^)]*\)\s*$", "", title).strip()
    w = re.sub(r"\s+people$", "", w)  # "Maasai people" -> "Maasai"
    return w


def article_block(title, text):
    text = text.strip()
    cut = re.split(r"\n==+ *(See also|References|External links|Notes|Further reading)", text)[0]
    return f'Title: {title}\nArticle text:\n"""\n{cut[:MAX_ARTICLE_CHARS]}\n"""'


def gen_messages(title, text):
    msgs = [{"role": "system", "content": SYSTEM_GEN}]
    for t, body, out in FEWSHOT:
        msgs.append({"role": "user", "content": article_block(t, body)})
        msgs.append({"role": "assistant", "content": json.dumps(out)})
    msgs.append({"role": "user", "content": article_block(title, text)})
    return msgs


def hint_problems(hint):
    """Return a list of plain-English problems the rewrite pass should fix."""
    probs = []
    if len(hint.split()) > 22:
        probs.append("it is too long: rewrite it in at most 18 words")
    if FILLER.search(hint):
        probs.append("it uses filler words (renowned, significant, leading, notable, "
                     "prominent): remove them")
    if STALE.search(hint):
        probs.append("it uses time-relative words (current, currently, latest, "
                     "recently): remove them and state only facts that stay true")
    return probs


def repair_messages(title, text, hint, problems):
    return [
        {"role": "system", "content": SYSTEM_REPAIR},
        {"role": "user", "content":
            f"{article_block(title, text)}\n\nHINT: {hint}\n\n"
            "Rewrite the hint as ONE plain sentence, fixing these problems: "
            + "; ".join(problems) + ". Keep it about why Kenyans know the subject "
            "or what it is popularly known for. Return ONLY the new sentence."},
    ]


def clean_sentence(raw):
    for line in raw.strip().splitlines():
        line = line.strip().strip('`"“”').strip()
        if line:
            line = " ".join(line.split())
            return line if len(line.split()) <= 60 else ""
    return ""


def check_messages(text, hint):
    return [
        {"role": "system", "content": SYSTEM_CHECK},
        {"role": "user", "content":
            f'Article text:\n"""\n{text.strip()[:MAX_ARTICLE_CHARS]}\n"""\n\nCLAIM: {hint}\n\n'
            "Is every factual statement in the CLAIM directly supported by the text? "
            "Answer with exactly one word: SUPPORTED or UNSUPPORTED."},
    ]


def parse_gen(raw):
    """Pull the JSON object out of the model's reply. Returns dict or None."""
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(d, dict):
        return None
    al = d.get("aliases", d.get("alias", []))
    if isinstance(al, str):
        al = [al]
    if not isinstance(al, list):
        al = []
    al = [str(a).strip() for a in al if str(a).strip()]
    return {
        "aliases": al,
        "hint": " ".join(str(d.get("hint", "")).split()),
        "hook": d.get("hook") is True,
    }


def parse_check(raw):
    r = raw.strip().upper()
    if r.startswith("UNSUPPORTED"):
        return False
    return r.startswith("SUPPORTED")


# --------------------------------------------------------------- the model ---
class QwenRunner:
    def __init__(self, model_name):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=quant, device_map="auto",
        )
        self.model.eval()

    def run(self, messages_list, max_new_tokens):
        prompts = [self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                   for m in messages_list]
        enc = self.tok(prompts, return_tensors="pt", padding=True).to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=self.tok.pad_token_id,
            )
        gen = out[:, enc["input_ids"].shape[1]:]
        return [self.tok.decode(g, skip_special_tokens=True) for g in gen]


class StubRunner:
    """No-GPU stand-in so the pipeline can be tested anywhere.
    Titles containing 'Long' get a long, filler-heavy hint so the repair pass runs."""
    def run(self, messages_list, max_new_tokens):
        outs = []
        for msgs in messages_list:
            head, last = msgs[0]["content"], msgs[-1]["content"]
            title = re.search(r"Title: (.*)", last).group(1) if "Title: " in last else "X"
            if "fact checker" in head:
                outs.append("SUPPORTED")
            elif "edit one-sentence clues" in head:
                outs.append(f"{title} is widely known across Kenya.")
            else:
                hint = (f"{title} is a renowned and significant subject " + "that Kenyans know well " * 4
                        if "Long" in title else f"{title} is widely known across Kenya.")
                outs.append(json.dumps({"aliases": ["GamingCompany", "Subject", "Thing", "Item"],
                                        "hook": True, "hint": hint}))
        return outs


# ------------------------------------------------------------------- main ---
def batches(items, n):
    for i in range(0, len(items), n):
        yield items[i:i + n]


def main():
    global MAX_ARTICLE_CHARS
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent.parent
    ap.add_argument("--articles", default=str(root / "data" / "raw" / "articles.jsonl"))
    ap.add_argument("--out", default=str(root / "data" / "processed" / "entries_raw.jsonl"))
    ap.add_argument("--prev", default="", help="earlier output to continue from (copied into --out first)")
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--max-chars", type=int, default=MAX_ARTICLE_CHARS)
    ap.add_argument("--limit", type=int, default=0, help="process only N new articles")
    ap.add_argument("--stub", action="store_true", help="use a fake model (no GPU)")
    a = ap.parse_args()
    MAX_ARTICLE_CHARS = a.max_chars

    arts = []
    with open(a.articles, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                arts.append(json.loads(line))

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if a.prev and not out_path.exists():
        shutil.copy(a.prev, out_path)
        print(f"Continuing from {a.prev}", flush=True)
    done = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            done = {json.loads(l)["title"] for l in f if l.strip()}
    todo = [x for x in arts if x["title"] not in done]
    if a.limit:
        todo = todo[:a.limit]
    print(f"{len(arts)} articles | {len(done)} done | {len(todo)} to process", flush=True)
    if not todo:
        return

    runner = StubRunner() if a.stub else QwenRunner(a.model)
    todo.sort(key=lambda x: len(x["text"][:MAX_ARTICLE_CHARS]))  # similar lengths per batch
    t0, n_done = time.time(), 0

    with open(out_path, "a", encoding="utf-8") as out:
        for chunk in batches(todo, a.batch_size):
            # Stage A: generate
            raws = runner.run([gen_messages(x["title"], x["text"]) for x in chunk], 200)
            recs = []
            for x, raw in zip(chunk, raws):
                p = parse_gen(raw)
                recs.append({
                    "title": x["title"], "url": x["url"], "group": x["group"],
                    "word": clean_word(x["title"]),
                    "aliases": p["aliases"] if p else [],
                    "alias": p["aliases"][0] if p and p["aliases"] else "",
                    "hint": p["hint"] if p else "",
                    "hook": bool(p and p["hook"]), "parsed": p is not None,
                    "repaired": False, "grounded": None, "raw": raw if p is None else "",
                })
            # Stage B: one rewrite pass for hints with fixable problems
            rep = [(r, x) for r, x in zip(recs, chunk)
                   if r["parsed"] and r["hook"] and r["hint"] and hint_problems(r["hint"])]
            if rep:
                fixed = runner.run([repair_messages(x["title"], x["text"], r["hint"],
                                                    hint_problems(r["hint"])) for r, x in rep], 80)
                for (r, _), o in zip(rep, fixed):
                    new = clean_sentence(o)
                    if new:
                        r["hint"], r["repaired"] = new, True
            # Stage C: grounding check on the final hint
            need = [(r, x) for r, x in zip(recs, chunk) if r["parsed"] and r["hook"] and r["hint"]]
            if need:
                checks = runner.run([check_messages(x["text"], r["hint"]) for r, x in need], 4)
                for (r, _), c in zip(need, checks):
                    r["grounded"] = parse_check(c)
            for r in recs:
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
            out.flush()
            n_done += len(chunk)
            rate = n_done / (time.time() - t0)
            print(f"{n_done}/{len(todo)}  ({rate:.2f} articles/s, "
                  f"~{(len(todo) - n_done) / max(rate, 1e-9) / 60:.0f} min left)", flush=True)
    print("Done:", out_path)


if __name__ == "__main__":
    main()
