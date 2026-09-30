from candor.load import load_all, parse_ts
from candor.retrieve import Index

store = load_all("data")
index = Index(store)

question = "Why did the launch slip from September 30?"
as_of = parse_ts("2026-09-18T18:00:00-07:00")
result = index.search(question, as_of, k=20)

for rank, hit in enumerate(result.hits, 1):
    print(f"\n#{rank}  score={hit.score:.3f}  id={hit.id}")
    print(f"source: {hit.record.source}")
    print(hit.text)