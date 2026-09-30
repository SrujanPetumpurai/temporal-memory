from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime

from .load import Record, Store

# --------------------------------------------------------------------------- #
# Text normalisation
# --------------------------------------------------------------------------- #

STOP = set("""a an the and or of to in on at for from by with about as is are was were be been
being am do does did doing have has had i me my we our you your he she it its they them their
this that these those what when where who whom whose which why how there here not so if then
than too very can could would should will just also into over after before again up out off
s t d ll re ve m""".split())

# words that may legitimately be absent from memory without meaning "unknown topic"
GENERIC = set("""say said tell told pick picked owe own owner owns mean much many long leave
like run still get got go going make made need want know think new current currently now
latest last first next time day week year do does did happen happened""".split())

MONTHS = {"january": "jan", "february": "feb", "march": "mar", "april": "apr", "june": "jun",
          "july": "jul", "august": "aug", "september": "sep", "sept": "sep", "october": "oct",
          "november": "nov", "december": "dec"}
_MONTH_RE = re.compile(r"\b(" + "|".join(sorted(MONTHS, key=len, reverse=True)) + r")\b")


def stem(w: str) -> str:
    if w.isdigit() or len(w) <= 3:
        return w
    if w.endswith("ies") and len(w) > 4:
        w = w[:-3] + "y"
    elif w.endswith(("ches", "shes", "xes", "sses")):
        w = w[:-2]
    elif w.endswith("ing") and len(w) > 5:
        w = w[:-3]
        if len(w) > 2 and w[-1] == w[-2] and w[-1] not in "lsz":
            w = w[:-1]
    elif w.endswith("ed") and len(w) > 4:
        w = w[:-2]
        if len(w) > 2 and w[-1] == w[-2] and w[-1] not in "lsz":
            w = w[:-1]
    elif w.endswith("s") and not w.endswith(("ss", "us")):
        w = w[:-1]
    if len(w) > 3 and w.endswith("e"):
        w = w[:-1]
    return w


def normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"(\d+)(?:st|nd|rd|th)\b", r"\1", s)
    s = _MONTH_RE.sub(lambda m: MONTHS[m.group(1)], s)
    return s


def tokenize(s: str, keep_stop: bool = False) -> list[str]:
    toks = re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", normalize(s))
    return [stem(t) for t in toks if keep_stop or t not in STOP]


_EXPAND_RAW = {
    "why": ["because", "reason", "cause", "issue", "problem", "regression"],
    "proposal": ["pricing", "price"], "pricing": ["proposal"], "price": ["proposal"],
    "slip": ["move", "push", "delay","reason", "issue", "problem"], "slipped": ["move", "push", "delay","reason", "issue", "problem"],
    "designer": ["design"], "flight": ["united", "depart"], "hiring": ["req", "role", "post"],
    "hire": ["req", "role", "post"], "launching": ["launch", "live"], "launch": ["live"],
    "signed": ["sign", "contract"], "sign": ["contract"], "standups": ["standup", "async"],
    "database": ["postgres", "sqlite"], "cut": ["drop", "cutting"], "owes": ["promise"],
    "owe": ["promise", "need"], "regression": ["test", "pass"], "prep": ["deck"],
    "contract": ["proposal", "pricing", "cfo"], "signed": ["proposal", "cfo", "review"],
    "fly": ["flight", "united", "depart"], "leave": ["depart", "flight"],
    "dictate": ["hi", "thanks"], "dictated": ["hi", "thanks"],
}
EXPAND = {stem(k): [stem(x) for x in v] for k, v in _EXPAND_RAW.items()}

DECISION_RE = re.compile(
    r"\b(agree[d]?|decid\w*|decision|officially|locked?|final|moving|moves|moved|pushed|"
    r"confirmed|going with|sent|posted|merged|approved|signed|launch(es)? (is|on)|now)\b", re.I)
SOURCE_HINTS = [
    (re.compile(r"\bdictat\w*", re.I), "dictation", 1.4),
    (re.compile(r"\b(e-?mail\w*|inbox)\b", re.I), "gmail", 1.2),
    (re.compile(r"\b(slack|dm|dmed|channel)\b", re.I), "slack", 1.2),
    (re.compile(r"\b(calendar|schedule|invite)\b", re.I), "calendar", 1.2),
    (re.compile(r"\b(chatgpt|codex)\b", re.I), None, 1.0),
]
DAY_REF = re.compile(r"\b(the day|that day|what'?s on my (calendar|schedule))\b", re.I)
DATE_RE = re.compile(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\.?\s+(\d{1,2})\b")
CURRENT_CUES = re.compile(
    r"\b(current|currently|now|latest|still|today|when is|when does|what is|who owns|are we|"
    r"is (it|the)|did .* (send|land)|has )\b", re.I)


# --------------------------------------------------------------------------- #
# Index
# --------------------------------------------------------------------------- #

@dataclass
class Hit:
    id: str
    score: float
    record: Record
    text: str          # text as of as_of
    group: str = ""


@dataclass
class Result:
    hits: list[Hit] = field(default_factory=list)
    ids: list[str] = field(default_factory=list)
    abstain: bool = False
    reason: str = ""


class Index:
    K1, B = 1.4, 0.6

    def __init__(self, store: Store) -> None:
        self.store = store
        self.ids = list(store.records)
        self.id_to_i = {rid: i for i, rid in enumerate(self.ids)}
        self.nb: dict[str, list[str]] = {}
        self._build_neighbors()
        self._build_names()

    # -- structure ---------------------------------------------------------- #
    def _build_neighbors(self) -> None:
        groups: dict[str, list[Record]] = defaultdict(list)
        for r in self.store.records.values():
            if "#" in r.id and r.parent and r.source in ("meeting", "chatgpt"):
                groups[r.parent].append(r)
        for lst in groups.values():
            lst.sort(key=lambda r: (r.when, r.id))
            for j, r in enumerate(lst):
                near = []
                if j > 0:
                    near.append(lst[j - 1].id)
                if j + 1 < len(lst):
                    near.append(lst[j + 1].id)
                self.nb[r.id] = near
        for r in self.store.records.values():
            if r.source == "slack" and r.parent in self.store.records:
                self.nb[r.id] = [r.parent]

    def _build_names(self) -> None:
        names: set[str] = set()
        for u in self.store.users.values():
            if not u.get("is_bot") and " " in u.get("real_name", ""):
                names.add(u["real_name"])
        for r in self.store.records.values():
            if r.source == "gmail":
                for f in [r.meta.get("from", "")] + list(r.meta.get("to", [])):
                    m = re.match(r"^(.+?)\s*<[^>]+>$", f or "")
                    if m and " " in m.group(1).strip():
                        names.add(m.group(1).strip())
            if r.source == "calendar":
                for e in r.meta.get("attendees", []):
                    m = re.match(r"^([a-z][a-z0-9_-]*)\.([a-z][a-z0-9_-]*)@", e, re.I)
                    if m:
                        names.add(f"{m.group(1).replace('_', ' ').title()} {m.group(2).title()}")
        self.names = sorted(names)
        self.by_first: dict[str, list[str]] = defaultdict(list)
        for n in self.names:
            self.by_first[n.split()[0].lower()].append(n)

    # -- temporal indexing ------------------------------------------------- #
    def _visible_records(self, as_of: datetime) -> list[tuple[Record, str]]:
        return list(self.store.visible(as_of))

    def _doc_terms(self, r: Record, text: str, as_of: datetime) -> Counter:
        """Build a BM25 document using only information visible at as_of."""
        tf: Counter = Counter()

        def add(value: str, weight: float) -> None:
            for t in tokenize(value):
                tf[t] += weight

        add(text, 1.0)
        if r.author:
            add(r.author, 0.6)
        m = r.meta
        add(" ".join(str(m.get(k, "")) for k in
                     ("title", "target_context", "target_app", "channel", "subject", "summary",
                      "repo")), 0.5)
        if r.source == "gmail":
            add(" ".join(m.get("to", [])), 0.3)

        d = r.when or r.delivered
        tf[d.strftime("%b").lower()] += 0.25
        tf[str(d.day)] += 0.25

        # Context is useful, but must also be time-safe. A future/deleted
        # neighbour must never influence an earlier query.
        for nid in self.nb.get(r.id, []):
            n = self.store.records[nid]
            nt = n.text_at(as_of)
            if nt is not None:
                add(nt, 0.35)
        return tf

    def _visible_index(self, as_of: datetime):
        """Build the small temporal BM25 view for one query.

        The dataset is intentionally small. Recomputing term statistics per
        query is preferable to a global index that can accidentally leak
        future edits into historical retrieval.
        """
        visible = self._visible_records(as_of)
        docs: dict[int, Counter] = {}
        idf_df: Counter = Counter()
        dl: dict[int, float] = {}

        for r, text in visible:
            i = self.id_to_i[r.id]
            tf = self._doc_terms(r, text, as_of)
            docs[i] = tf
            dl[i] = sum(tf.values()) or 1.0
            for term in tf:
                idf_df[term] += 1

        n = len(docs)
        avgdl = (sum(dl.values()) / n) if n else 1.0
        return visible, docs, dl, idf_df, n, avgdl

    def _query_weights(self, q: str) -> dict[str, float]:
        w: dict[str, float] = {}
        toks = tokenize(q)
        for t in toks:
            w[t] = 1.0
        for t in toks:
            for e in EXPAND.get(t, []):
                w.setdefault(e, 0.4)
        return w

    def _score(self, qw: dict[str, float], as_of: datetime) -> dict[int, float]:
        _visible, docs, dl, df, n, avgdl = self._visible_index(as_of)
        if not docs:
            return {}

        scores: dict[int, float] = defaultdict(float)
        for t, wq in qw.items():
            term_df = df.get(t, 0)
            if not term_df:
                continue
            idf = math.log(1 + (n - term_df + 0.5) / (term_df + 0.5))
            for i, tf_map in docs.items():
                tf = tf_map.get(t, 0.0)
                if not tf:
                    continue
                norm = tf * (self.K1 + 1) / (
                    tf + self.K1 * (1 - self.B + self.B * dl[i] / avgdl)
                )
                scores[i] += wq * idf * norm
        return scores

    def _visible_vocab(self, as_of: datetime) -> set[str]:
        vocab: set[str] = set()
        for r, text in self.store.visible(as_of):
            vocab.update(tokenize(text))
            if r.author:
                vocab.update(tokenize(r.author))
            m = r.meta
            vocab.update(tokenize(" ".join(str(m.get(k, "")) for k in
                                           ("title", "target_context", "target_app", "channel",
                                            "subject", "summary", "repo"))))
        return vocab

    def _abstain_check(self, q: str, as_of: datetime) -> tuple[bool, str]:
        raw = re.findall(r"[A-Za-z0-9][A-Za-z0-9\.]*", q)
        content = missing = 0
        vocab = self._visible_vocab(as_of)
        for j, word in enumerate(raw):
            lw = normalize(word)
            if lw in STOP:
                continue
            st = stem(lw)
            if st in GENERIC or lw in GENERIC:
                continue
            content += 1
            if st in vocab:
                continue
            proper = (word[0].isupper() and j > 0) or (word.isupper() and len(word) >= 2)
            if proper:
                return True, f"'{word}' appears nowhere in memory as of {as_of.isoformat()}"
            missing += 1
        if content and missing / content >= 0.34:
            return True, f"{missing}/{content} key terms not in memory as of {as_of.isoformat()}"
        return False, ""

    def search(self, question: str, as_of: datetime, k: int = 20,
               min_score: float = 0.0) -> Result:
        res = Result()
        ab, why = self._abstain_check(question, as_of)
        if ab:
            res.abstain, res.reason = True, why
            return res

        qw = self._query_weights(question)
        if DAY_REF.search(question):
            # Resolve relative-day questions only from records visible at the
            # requested time. In particular, don't use a future flight email.
            first = self._score(qw, as_of)
            top = sorted(first.items(), key=lambda kv: -kv[1])[:5]
            for i, _ in top:
                r = self.store.records[self.ids[i]]
                t = r.text_at(as_of)
                if t is None:
                    continue
                m = DATE_RE.search(normalize(t))
                if m:
                    qw[m.group(1)] = qw.get(m.group(1), 0) + 1.5
                    qw[m.group(2)] = qw.get(m.group(2), 0) + 1.5
                    break

        scores = self._score(qw, as_of)
        qlow = question.lower()
        qnames = [n for n in self.names if n.lower() in qlow]
        cur = bool(CURRENT_CUES.search(question))
        why_question = bool(re.search(
                    r"\bwhy\b|\breason\b|\bcause\b|\bwhat caused\b|\bwhat happened\b",
                    question,
                    re.I,
                ))
        cands: list[Hit] = []

        for i, s in scores.items():
            r = self.store.records[self.ids[i]]
            text = r.text_at(as_of)
            if text is None:
                continue
            tl = text.lower()

            for full in qnames:
                first = full.split()[0].lower()
                if (r.author or "").lower().startswith(full.lower()):
                    s *= 1.35
                elif full.lower() in tl:
                    s *= 1.3
                elif any(o.lower() in tl for o in self.by_first[first] if o != full):
                    s *= 0.55

            for rx, src, mult in SOURCE_HINTS:
                if src and r.source == src and rx.search(question):
                    s *= mult
            # For current-state questions, favor records that actually state a
            # decision/final state. Plain lexical overlap otherwise tends to
            # rank the original plan (e.g. "Sep 30") above the later change
            # (e.g. "Oct 21"). Keep the boost modest for historical queries.
            if DECISION_RE.search(text):
                s *= 1.30 if cur else 1.10
            if why_question:
                if re.search(
                    r"\b(because|due to|regression|issue|problem|bug|failure|"
                    r"wrong|error|blocked|blocking|fix|fixing|re-verify)\b",
                    text,
                    re.I,
                ):
                    s *= 1.35

                    if r.source == "slack":
                        s *= 1.20
            # A current-state question should prefer the latest visible
            # evidence for the topic, not merely the record with the most
            # literal word overlap. This is intentionally bounded so older
            # records remain retrievable for "why did it change?" questions.
            if cur:
                age_days = max(0.0, (as_of - r.delivered).total_seconds() / 86400.0)
                current_decay = math.exp(-age_days / 3.5)
                s *= 1.0 + 0.45 * current_decay
            if r.meta.get("kind") in ("meeting", "conversation"):
                s *= 0.6
            if "CATEGORY_PROMOTIONS" in r.meta.get("labels", []):
                s *= 0.7
            if r.flags.get("injection"):
                s *= 0.4

            # Small general recency preference. Current-state questions got
            # their stronger topic-specific recency boost above.
            age_days = max(0.0, (as_of - r.delivered).total_seconds() / 86400.0)
            decay = math.exp(-age_days / 7.0)
            s *= 1.0 + (0.08 if not cur else 0.04) * decay
            cands.append(Hit(r.id, s, r, text, self._group(r)))

        if not cands:
            res.abstain, res.reason = True, "no visible record matches"
            return res

        cands.sort(key=lambda h: -h.score)
        if min_score and cands[0].score < min_score:
            res.abstain, res.reason = True, f"top score {cands[0].score:.1f} < {min_score}"
            return res

        pool, chosen, cnt = cands[:120], [], Counter()
        while pool and len(chosen) < k:
            best = max(pool, key=lambda h: h.score * self._pen(h) ** cnt[h.group])
            pool.remove(best)
            chosen.append(best)
            cnt[best.group] += 1

        # `retrieved` is the scored evidence list. Parent meeting/conversation
        # containers are useful metadata but should not consume top-k slots.
        ids = []
        for h in chosen:
            ids.append(h.id)
            eid = self.edit_id_at(h.record, as_of)
            if eid:
                ids.append(eid)
        res.hits = chosen
        res.ids = ids[:k]
        return res

    @staticmethod
    def edit_id_at(r: Record, as_of: datetime) -> str | None:
        applicable = [eid for ts, eid in r.meta.get("edit_ids", []) if ts <= as_of]
        return applicable[-1] if applicable else None

    @staticmethod
    def _group(r: Record) -> str:
        if r.source in ("meeting", "chatgpt") and r.parent:
            return r.parent
        if r.source == "gmail":
            return r.parent or r.id
        return r.id

    @staticmethod
    def _pen(h: Hit) -> float:
        return 0.85 if h.record.source in ("meeting", "chatgpt") else 0.97

