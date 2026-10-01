import json
from candor.load import load_all, parse_ts
from candor.retrieve import Index
ix = Index(load_all("data"))
for q in map(json.loads, open("evals/memory_train.jsonl")):
    if q["id"] in ("MEM-TR-20", "MEM-TR-21"):
        r = ix.search(q["question"], parse_ts(q["as_of"]))
        print(q["id"], "needed:", q["needed"])
        for h in r.hits[:12]:
            print(f"  {h.score:6.2f} {h.id}  {h.text[:70]!r}")