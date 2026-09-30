from __future__ import annotations
 
import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Iterator
 
# --------------------------------------------------------------------------- #
# Safety filters
# --------------------------------------------------------------------------- #
 
SECRET_PATTERNS = [
    re.compile(r"\b(?:sk|pk|rk|ghp|xox[abp])-[A-Za-z0-9_\-]{12,}"),
    re.compile(r"(?i)\b(?:api[_ -]?key|password|passwd|secret|token)\s*[:=]\s*\S{6,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
]
INJECTION_PATTERNS = [
    re.compile(r"<!--.*?-->", re.S),  # hidden HTML comments
    re.compile(r"(?i)(ignore (all |your )?(previous|prior) instructions[^.\n]*[.\n]?)"),
    re.compile(r"(?i)(note to (any )?ai assistant[^.\n]*[.\n]?)"),
]
 
 
def redact_secrets(text: str) -> tuple[str, bool]:
    hit = False
    for pat in SECRET_PATTERNS:
        text, n = pat.subn("[REDACTED_SECRET]", text)
        hit = hit or n > 0
    return text, hit
 
 
def neutralize_injection(text: str) -> tuple[str, bool]:
    """Instructions inside data are content, never commands: cut them out."""
    hit = False
    for pat in INJECTION_PATTERNS:
        text, n = pat.subn("[planted instruction removed]", text)
        hit = hit or n > 0
    return text, hit
 
 
def clean(text: str) -> tuple[str, dict]:
    text, inj = neutralize_injection(text or "")
    text, sec = redact_secrets(text)
    flags = {}
    if inj:
        flags["injection"] = True
    if sec:
        flags["secret"] = True
    return text, flags
 
 
# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
 
def parse_ts(s: str | None) -> datetime | None:
    if not s:
        return None
    if len(s) == 10:  # all-day "YYYY-MM-DD"
        s += "T00:00:00-07:00"
    return datetime.fromisoformat(s.replace("Z", "+00:00"))
 
 
@dataclass
class Record:
    id: str
    source: str                 # meeting | dictation | slack | gmail | calendar | codex | chatgpt
    delivered: datetime         # exists from this moment on
    text: str                   # current (latest) cleaned text
    author: str | None = None   # who produced it (display name / email)
    author_confidence: float | None = None
    parent: str | None = None   # meeting id, conversation id, thread id, ...
    when: datetime | None = None  # event time (may differ from delivery)
    meta: dict = field(default_factory=dict)
    versions: list[tuple[datetime, str]] = field(default_factory=list)  # (from_ts, text)
    deleted_at: datetime | None = None
    flags: dict = field(default_factory=dict)
 
    def text_at(self, as_of: datetime) -> str | None:
        """Text as it read at `as_of`, or None if not yet delivered / deleted."""
        if self.delivered > as_of:
            return None
        if self.deleted_at and self.deleted_at <= as_of:
            return None
        text = self.text
        if self.versions:
            text = self.versions[0][1]
            for ts, t in self.versions:
                if ts <= as_of:
                    text = t
        return text
 
 
class Store:
    def __init__(self) -> None:
        self.records: dict[str, Record] = {}
        self.users: dict[str, dict] = {}
        self.channels: dict[str, dict] = {}
 
    def add(self, r: Record) -> None:
        self.records[r.id] = r
 
    def __len__(self) -> int:
        return len(self.records)
 
    def visible(self, as_of: datetime) -> Iterator[tuple[Record, str]]:
        """Yield (record, text-as-of) for everything that exists at `as_of`."""
        for r in self.records.values():
            t = r.text_at(as_of)
            if t is not None:
                yield r, t
 
 
# --------------------------------------------------------------------------- #
# File discovery
# --------------------------------------------------------------------------- #
 
def read_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))
 
 
def read_jsonl(p: Path) -> list[dict]:
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out
 
 
def find(root: Path, pattern: str) -> list[Path]:
    return sorted(p for p in root.rglob(pattern) if p.is_file())
 
 
def sniff_jsonl(paths: Iterable[Path]) -> dict[str, list[Path]]:
    """messages.jsonl exists for both Slack and Gmail: tell them apart by fields."""
    kinds: dict[str, list[Path]] = {"slack": [], "gmail": []}
    for p in paths:
        with p.open(encoding="utf-8") as fh:
            first = fh.readline()
        if not first.strip():
            continue
        row = json.loads(first)
        if "channel_id" in row:
            kinds["slack"].append(p)
        elif "thread_id" in row:
            kinds["gmail"].append(p)
    return kinds
 
 
# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
 
def load_meetings(store: Store, root: Path) -> None:
    for p in find(root, "MTG-*.json"):
        m = read_json(p)
        start = parse_ts(m["start"])
        seg_ids = []
        for s in m["segments"]:
            text, flags = clean(s["text"])
            name = s.get("speaker_name")
            store.add(Record(
                id=s["seg_id"], source="meeting", text=text,
                delivered=start + timedelta(seconds=s["end_s"]),
                when=start + timedelta(seconds=s["start_s"]),
                author=name or f"unidentified ({s['speaker_label']})",
                author_confidence=s.get("speaker_confidence"),
                parent=m["id"], flags=flags,
                meta={"title": m["title"], "speaker_label": s["speaker_label"],
                      "identified": name is not None, "channel": s.get("channel"),
                      "start_s": s["start_s"], "end_s": s["end_s"]},
            ))
            seg_ids.append(s["seg_id"])
        # whole-meeting record (only counts as "found the right meeting")
        store.add(Record(
            id=m["id"], source="meeting", text=m["title"],
            delivered=start + timedelta(seconds=m["segments"][-1]["end_s"]),
            when=start, parent=None,
            meta={"kind": "meeting", "title": m["title"], "type": m.get("type"),
                  "end": m["end"], "calendar_event_id": m.get("calendar_event_id"),
                  "participants": m.get("participants_known", []), "segments": seg_ids},
        ))
 
 
def load_dictations(store: Store, root: Path) -> None:
    for p in find(root, "dictations.jsonl"):
        for d in read_jsonl(p):
            text, flags = clean(d["cleaned_text"])
            ts = parse_ts(d["timestamp"])
            store.add(Record(
                id=d["id"], source="dictation", delivered=ts, when=ts, text=text,
                author="Alex Rivera", flags=flags,
                meta={"mode": d["mode"], "target_app": d["target_app"],
                      "target_context": d["target_context"],
                      "delivery_state": d["delivery_state"]},
            ))
 
 
def load_slack(store: Store, root: Path, paths: list[Path]) -> None:
    for p in find(root, "users.json"):
        for u in read_json(p):
            store.users[u["id"]] = u
    for p in find(root, "channels.json"):
        for c in read_json(p):
            store.channels[c["id"]] = c
 
    events = []
    for p in paths:
        for m in read_jsonl(p):
            if m.get("subtype") in ("message_changed", "message_deleted"):
                events.append(m)
                continue
            u = store.users.get(m["user"], {})
            ch = store.channels.get(m["channel_id"], {})
            name = m.get("bot_name") or u.get("real_name") or m["user"]
            text, flags = clean(m["text"])
            ts = parse_ts(m["ts"])
            store.add(Record(
                id=m["id"], source="slack", delivered=ts, when=ts, text=text,
                author=name, parent=m.get("thread_parent_id"), flags=flags,
                meta={"channel_id": m["channel_id"], "channel": ch.get("name"),
                      "is_dm": ch.get("is_dm", False), "user": m["user"],
                      "bot": m.get("subtype") == "bot_message",
                      "reactions": m.get("reactions", [])},
            ))
    # apply edits / deletions in time order
    for ev in sorted(events, key=lambda e: e["ts"]):
        target = store.records.get(ev["target_id"])
        if not target:
            continue
        ts = parse_ts(ev["ts"])
        if ev["subtype"] == "message_changed":
            new, flags = clean(ev["text"])
            if not target.versions:
                target.versions.append((target.delivered, target.text))
            target.versions.append((ts, new))
            target.meta.setdefault("edit_ids", []).append((ts, ev["id"]))
            target.text = new
            target.flags.update(flags)
        else:
            target.deleted_at = ts
            target.flags["deleted"] = True
            target.meta["delete_id"] = ev["id"]
    # never keep a deleted message's text around for downstream use
    for r in store.records.values():
        if r.deleted_at:
            r.meta["_deleted_text_dropped"] = True
 
 
def load_gmail(store: Store, paths: list[Path]) -> None:
    for p in paths:
        for e in read_jsonl(p):
            ts = parse_ts(e["date"])
            body, flags = clean(e["body"])
            sender = e["from"]
            store.add(Record(
                id=e["id"], source="gmail", delivered=ts, when=ts,
                text=f"Subject: {e['subject']}\n{body}", author=sender,
                parent=e["thread_id"], flags=flags,
                meta={"from": sender, "to": e["to"], "cc": e.get("cc", []),
                      "subject": e["subject"], "labels": e.get("labels", []),
                      "attachments": [a["filename"] for a in e.get("attachments", [])]},
            ))
 
 
def _cal_time(d: dict) -> str | None:
    return d.get("dateTime") or d.get("date")
 
 
def load_calendar(store: Store, root: Path) -> None:
    for p in find(root, "events.jsonl"):
        for ev in read_jsonl(p):
            if "summary" not in ev:
                continue
            upd = parse_ts(ev.get("updated") or ev.get("created"))
            start, end = _cal_time(ev["start"]), _cal_time(ev["end"])
            attendees = ", ".join(
                f"{a['email']} ({a['responseStatus']})" for a in ev.get("attendees", []))
            text, flags = clean(
                f"{ev['summary']} | {start} to {end} | {ev.get('location', '')} | "
                f"status={ev['status']} | organizer={ev['organizer']} | "
                f"{ev.get('description', '')} | attendees: {attendees}")
            store.add(Record(
                id=ev["id"], source="calendar", delivered=upd, text=text,
                when=parse_ts(start), author=ev["organizer"], flags=flags,
                meta={"summary": ev["summary"], "start": start, "end": end,
                      "status": ev["status"], "recurrence": ev.get("recurrence"),
                      "all_day": "date" in ev["start"],
                      "attendees": [a["email"] for a in ev.get("attendees", [])]},
            ))
 
 
def load_codex(store: Store, root: Path) -> None:
    for p in find(root, "CDX-*.jsonl"):
        rows = read_jsonl(p)
        meta = next((r for r in rows if r["type"] == "session_meta"), {})
        events = [r for r in rows if r["type"] != "session_meta"]
        parts, last = [], parse_ts(meta.get("started_at"))
        for r in events:
            ts = parse_ts(r["timestamp"])
            last = max(last, ts) if last else ts
            if r["type"] == "message":
                parts.append(f"{r['role'].upper()}: {r['content']}")
            else:  # tool call: keep command + short output
                out = (r.get("output") or "")[:400]
                parts.append(f"TOOL {r['tool']}: {r['input'][:300]}\n-> {out}")
        text, flags = clean("\n".join(parts))
        sid = meta.get("id", p.stem)
        store.add(Record(
            id=sid, source="codex", delivered=last, text=text, author="Alex Rivera",
            when=parse_ts(meta.get("started_at")), flags=flags,
            meta={"repo": meta.get("repo"), "cwd": meta.get("cwd")},
        ))
 
 
def load_chatgpt(store: Store, root: Path) -> None:
    for p in find(root, "conversations.json"):
        for conv in read_json(p):
            store.add(Record(
                id=conv["id"], source="chatgpt",
                delivered=parse_ts(conv["messages"][0]["create_time"]),
                when=parse_ts(conv["create_time"]), text=conv["title"],
                meta={"kind": "conversation", "title": conv["title"],
                      "updated": conv["update_time"]},
            ))
            for m in conv["messages"]:
                text, flags = clean(m["content"])
                ts = parse_ts(m["create_time"])
                store.add(Record(
                    id=m["id"], source="chatgpt", delivered=ts, when=ts, text=text,
                    author="Alex Rivera" if m["role"] == "user" else "ChatGPT",
                    parent=conv["id"], flags=flags,
                    meta={"role": m["role"], "title": conv["title"]},
                ))
 
 
# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #
 
def load_all(data_dir: str | Path = "data") -> Store:
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"data dir not found: {root}")
    store = Store()
    load_meetings(store, root)
    load_dictations(store, root)
    kinds = sniff_jsonl(find(root, "messages.jsonl"))
    load_slack(store, root, kinds["slack"])
    load_gmail(store, kinds["gmail"])
    load_calendar(store, root)
    load_codex(store, root)
    load_chatgpt(store, root)
    return store
 
 
def main() -> None:
    ap = argparse.ArgumentParser(description="Load Candor data and print a summary")
    ap.add_argument("--data", default="data")
    ap.add_argument("--as-of", help="ISO time; count only records visible then")
    args = ap.parse_args()
    store = load_all(args.data)
    print(f"loaded {len(store)} records")
    print("by source:", dict(Counter(r.source for r in store.records.values())))
    flagged = Counter(k for r in store.records.values() for k in r.flags)
    print("flags:", dict(flagged))
    if args.as_of:
        t = parse_ts(args.as_of)
        n = sum(1 for _ in store.visible(t))
        print(f"visible at {args.as_of}: {n}")
 
 
if __name__ == "__main__":
    main()
 