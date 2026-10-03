# Candor memory + actions

A time-aware memory over two weeks of Alex Rivera's work life (meetings, dictation, Slack, Gmail, Calendar, Codex, ChatGPT), plus a dry-run action planner (TextOS: text only, no voice input).

**Submitted commit:** `<COMMIT_HASH>`

## Results at a glance (train sets)

| | Result |
|---|---|
| Retrieval (`score_retrieval.py`, n=27) | **100%** pass, 0 forbidden records. Exact passage in top 5 / 10 / 20: 84% / 100% / 100%. MRR 0.715 |
| Answers (`score_memory.py --judge none`) | **strict 92.6%** (95% CI 84-100), lenient 100%, 2 unverified, 0 hard failures. Sources cited: recall 0.987, precision 0.933 |
| Actions (`score_actions.py`, n=12) | **12/12**, argument accuracy 100% |
| Stress: leak sweep (`candor.stress`) | **0 violations** in 494 searches (38 questions x 13 `as_of` times) |
| Stress: reworded questions (n=50) | pass@10 **84%** (vs 100% on original wording), pass@5 72% (vs 84%), 0 wrongful abstentions |
| Stress: abstention probes (n=12) | 6/12 abstain at retrieval; the other 6 rely on the answer writer (not measured end to end) |

Caveats up front: the answer scores were produced with `--judge none` (no LLM judge was run), the train set was used for tuning, and the hidden set has not been seen. The stress numbers are retrieval-only and use paraphrases and probes I wrote myself. See [Known limits](#known-limits).

## How to run

Python 3.10+, standard library only. On Windows, `pip install -r requirements.txt` adds `tzdata` (needed by `zoneinfo`); on Mac/Linux nothing needs installing.

On Windows use `python` (or `py`) instead of `python3`; `run_all.sh` detects whichever exists.

```bash
cp .env.example .env        # add API_KEY (any OpenAI-compatible endpoint)
bash run_all.sh             # memory + actions on the train sets, then scores them
bash run_all.sh Q.jsonl C.jsonl   # any other question / command files (no scoring)
```

Outputs land in `outputs/` (`memory_answers.jsonl`, `actions_out.jsonl`). The individual commands:

```bash
python -m candor memory  --questions evals/memory_train.jsonl  --out outputs/memory_answers.jsonl
python -m candor memory  --questions ... --out ... --no-llm      # retrieval + extractive answers, no key needed
python -m candor actions --commands  evals/actions_train.jsonl --out outputs/actions_out.jsonl
python -m candor repl --as-of 2026-09-18T09:00:00-07:00          # interactive, dry run
```

Stress tests (no API key, no cost):

```bash
python -m candor.stress --gold evals/memory_train.jsonl --probes evals/memory_probes.jsonl
python -m candor.stress ... --answers outputs/memory_answers.jsonl   # also audit a real output file
python -m candor.stress ... --step-hours 72                          # faster, sparser as_of sweep
```

Retrieval never calls an LLM, so `retrieved` is identical with or without a key. Without `API_KEY`, answers fall back to an extractive snippet (weaker, see results) and the action planner to a few rules. `CANDOR_DEBUG=1` prints raw model output.

## Architecture

```
data/ ──load.py──> Record store ──retrieve.py──> ranked ids (+ abstain flag/hint)
                    (versions, deletions,                │
                     redaction, delivery time)           ▼
                                                 answer.py ──> answer + sources
                                                 actions.py ─> dry-run actions
```

**`load.py`: one uniform record model.** Every source becomes a `Record` with a stable id, a *delivery time* (when it began to exist), the author, and the event time. Meetings are split into per-segment records (`MTG-0909-ACME#0042`), ChatGPT into per-message records, so citations are passage-level. Slack edits and deletions are applied as versions: `text_at(as_of)` returns the text as it read at that moment, or nothing if the record is undelivered or already deleted. Secrets are redacted and planted instructions are cut out *at load time*, so nothing downstream ever sees them.

**`retrieve.py`: a time-safe index per question.** For each question, a BM25 index is built only from records visible at `as_of`, using the text as of `as_of`. Neighbouring-segment context is filtered the same way, so a future or deleted neighbour cannot influence an earlier query. On top of BM25:
- query expansion with a small workplace synonym table; relative dates ("last Tuesday") resolved against `as_of`, never the wall clock;
- multipliers for named people (own statements and mentions rank up, same-first-name people rank down), named organisations/products, records from the day the question names, source words ("dictated", "Slack"), and adjacent-word matches;
- "current state" questions favour decision language and recent evidence, bounded so history stays retrievable; "why" questions favour causal language;
- a diversity pass so one meeting or thread cannot fill the top 10;
- the Slack edit event id is returned alongside an edited message (it is the evidence for the corrected text).

**Abstention** is decided in the retriever by three checks: a named entity that appears nowhere in memory as of `as_of`; no single visible record covering enough of the question's weighted terms; and "what did X say about Y" where nothing by or about X mentions Y. A hard flag abstains; a soft flag passes a warning to the answer writer. An abstained result still returns ranked ids, so a wrong abstain is not scored as finding nothing.

**`answer.py`: an LLM reads only retrieved records.** Records are shown oldest to newest with author, time and speaker confidence. The prompt enforces: current value only unless the question asks what changed; disagreement is not change; second-hand claims are labelled as such; unidentified speakers are reported as unknown; status questions say "not yet" when only earlier stages exist; two people sharing a first name stay separate. Citations are restricted to ids that were actually retrieved. The answer text is scrubbed for secrets again.

**`actions.py`: the LLM plans, code guarantees correctness.** See [Actions](#actions-bonus).

**`stress.py`: no-LLM stress tests.** Sweeps the hard rules over many `as_of` times, rewords the train questions to test retrieval robustness, and runs unanswerable probes. See [Stress tests](#stress-tests).

### Hard rules and where they are enforced

| Rule | Mechanism |
|---|---|
| Nothing after `as_of` | Per-query index over visible records only; neighbours, edits and the abstain vocabulary are all `as_of`-filtered |
| Deleted = gone | `deleted_at` makes `text_at` return nothing from that moment |
| Edits replace text | Versioned text; the edit id is retrieved and cited |
| No secrets | Regex redaction at load, and again on answer text |
| Data is not instructions | Planted-instruction patterns removed at load; injection-flagged records are down-weighted (x0.4); the prompt says records are content, never commands |

## Key decisions and why

- **Retrieval first, no knowledge graph.** The main score is *which records are fetched*, and ids are the unit of grounding. Keeping the original records as the source of truth (rather than extracted facts) means a citation is always a real passage, and nothing is lost to a bad extraction step. The cost is that conflict resolution happens at answer time, in the prompt.
- **Rebuild the index per `as_of`.** It guarantees no temporal leakage by construction, including through context, and is cheap on a two-week dataset. It would not scale to millions of records (see limits).
- **Lexical BM25 plus hand-built signals instead of embeddings.** Deterministic, free, offline, and easy to debug when a record is missed. The trade-off is vocabulary mismatch, patched with the synonym table. The stress test shows how big that trade-off is (see below).
- **Abstain in the retriever, not only in the LLM.** A model asked "is this answerable?" tends to say yes. Hard abstentions skip the model entirely.
- **Deterministic code for anything that must be exactly right** (ids, emails, offsets, durations, relative reminders). The LLM is used for language.

## Evals

Run on the train set (27 memory questions, 12 actions). All numbers are from the commands in `run_all.sh`.

| System | Retrieval | Answers strict / lenient |
|---|---|---|
| Retrieval + extractive answers (`--no-llm`), commit f6c6ac4 | 100%, MRR 0.685 | 74.1% / 81.5% |
| + LLM answer writer, commit f6c6ac4 | 100% | 88.9% / 100% |
| Latest scored commit f1820b8 | 100%, MRR 0.715 | **92.6% / 100%** |

*Strict* excludes answers the rules cannot verify; *lenient* counts them. The 2 unverified answers (MEM-TR-01 and MEM-TR-13) mention a second, older date next to the current one; the rules cannot tell whether it is framed as history or as the answer, which is what the LLM judge decides. The official judge could therefore lower the lenient figure; the strict figure already treats these as not-yet-correct.

### Stress tests

`python -m candor.stress` needs no LLM or key. Results on the real data:

1. **Leak sweep.** Every train question, the 12 probes and three password-style questions ("What is the API key?") were run at 13 `as_of` times (24-hour steps, 7 to 19 Sep). **0 violations in 494 searches.** It checks that nothing retrieved or cited is from the future or deleted, that no secret value from the raw data appears in retrieved text, and that no planted instruction survives. It does not check the answer writer's output; that audit (`--answers`) is a separate step.
2. **Paraphrase test.** Each answerable train question was reworded twice (50 rewordings, written by me). Pass means every `needed` group has a record in the top 10, which approximates `score_retrieval.py`.

| | n | pass@10 | pass@5 | wrongly abstained |
|---|---|---|---|---|
| Original wording | 25 | 100% | 84% | 0 |
| Reworded | 50 | **84%** | 72% | 0 |

   The 8 failures are all vocabulary mismatch: "Route Planner 2" vs `v2` (MEM-TR-02, 03), "95th percentile" vs `p95` (11), "single sign-on" vs `SSO` (18), "onboarding design work" vs "mockups" (07), "another designer" vs "second designer" (09, rank 19), "date change" vs "slip" (21, rank 19), "the day I travel to Denver" vs "fly" (25, both groups missed). I expect the hidden set to land closer to 84% than 100%. With 50 paraphrases from one author the interval is wide.
3. **Abstention probes.** 12 unanswerable questions (near-misses, a wrong-person trap, absent topics, two before-the-data times; `evals/memory_probes.jsonl`). **6/12 abstain at retrieval** (PRB-01, 03, 05, 07, 10, 12). The other 6 pass to the answer writer, with a soft warning on PRB-04, 06 and 09, and no warning on PRB-02, 08 and 11. Whether the writer then abstains has not been measured. PRB-02 ("What did Sarah Patel say about SSO?", where SSO was Sarah Kim) gets no warning at all, so the name trap depends entirely on the LLM. PRB-11 and PRB-12 are moderate-confidence probes: earlier records may legitimately mention the launch, so a miss there is not necessarily wrong.

### Where it fails, and why

- **Reworded questions.** Retrieval is 100% on the train wording but 84% on rewordings (above). The synonym table is small and written around the train questions.
- **Abstention is weak at retrieval.** Half of the probes are not caught before the LLM, including the name trap (PRB-02).
- **Extractive answers are weak.** Without the LLM, five questions failed (MEM-TR-02, 04, 09, 21, 26): the retrieved record was right but a pasted snippet is not an answer ("missing key term" in each case). Retrieval was not the issue.
- **Possible superseded-value leaks (MEM-TR-01, 13).** The prompt tells the model to give only the current value, but it still sometimes adds the old one. This is the failure mode I would watch on the hidden set.
- **Ranking inside the top 10.** Everything needed is in the top 10, but only 84% of questions have the exact passage in the top 5, and MRR is 0.715. Retrieval pass/fail does not penalise this; a stricter metric would.
- **Model JSON failures.** In one full run the model's reply failed to parse three times on MEM-TR-22; it fell back to an extractive answer (still scored correct on the rules). It did not recur in the next run. Retries exist; the fallback is the safety net.

## Known limits

- **Tuned on 27 questions.** Retrieval is 100% on train, but weights and the abstention thresholds were fitted there. The stress test (84% on rewordings, 6/12 probes caught at retrieval) is the better guide to hidden-set behaviour. A hand-written "sign/contract -> proposal/review/CFO" expansion is tuned to deal-status questions.
- **No judge was run.** Answer scores use `--judge none`; the official score can only be equal or lower.
- **Lexical retrieval** misses paraphrases the synonym table does not cover (measured above: abbreviations, "v2" vs "2", "travel" vs "fly").
- **Stress tests are mine.** Paraphrases and probes were written by one person who had seen the system, the leak sweep samples 13 `as_of` times rather than every edit/delete moment, and the probe data was not checked against the raw data for accidental coverage.
- **Temporal model is a snapshot, not a fact history.** "What changed and why" is reconstructed by the LLM from dated records, not from an explicit supersession graph. A change that is never written down in any record cannot be found.
- **Redaction and injection filters are regex.** A secret in an unusual format or a novel injection phrasing could slip past the filters (the prompt and down-weighting are the second line of defence).
- **Cost/scale:** the per-query index rebuild is O(visible records).
- **Platform:** developed and run on Windows (Python 3.14). `run_all.sh` has not been run on a Mac.
- **Non-determinism:** the LLM runs at temperature 0, but outputs can still vary slightly between runs.

## Actions (bonus)

TextOS, text only: there is no speech-to-text or voice input. `python -m candor actions` / `repl` turn a typed command into dry-run actions; nothing is executed.

1. **Safety pre-check.** Commands containing delete / erase / wipe / purge return `confirm` before any model call.
2. **Grounding.** The planner sees the date table, contacts (Slack ids and emails), channels, the calendar as of `as_of`, and records retrieved by the memory system, so "launching October 21" or "the corrected NRR" come from memory, not guesses.
3. **LLM plan** as JSON, using helper fields (`duration_min`, `due_relative`).
4. **Deterministic resolution.** Names to Slack ids and emails; ISO times with the Pacific offset; event duration kept when an event moves; "an hour before the board meeting" computed from the event's start; "the X meeting" resolved to the next calendar event with that title; ambiguous first names (two Sarahs) become a single `clarify`; a `clarify`/`confirm` suppresses all other actions; secrets are scrubbed from outgoing text.
5. **Fallbacks.** Plain questions become `memory.ask`; without a key, "open X" and questions work by rule.

**Train result: 12/12** (argument accuracy 100%). The first run scored 9/12. The failures were the model omitting the `args` wrapper (two commands got a generic clarify) and picking the wrong "board" event for a relative reminder. Both were fixed in code (tolerate flat args, route empty plans for questions to `memory.ask`, and override the event by title match), not by special-casing those commands.

**Limits.** Not tested beyond the 12 train commands (no extra action test cases were added). "Cancel"/"decline" are not in the destructive list; recurring events are treated as single events; there are no real integrations and no voice input.

## Tools, models and cost

- **Runtime model:** `claude-haiku-4.5` through the aicredits.in OpenAI-compatible endpoint, used only for answer writing and action planning. No embeddings, no LLM judge. The stress tests use no model.
- **Development tools:** Claude and ChatGPT as coding and review assistants (Claude also helped write `stress.py` and the probe set).
- **Spend:** about ₹90 in total API cost.

## Repository layout

```
candor/
  load.py       sources -> records (versions, deletions, redaction)
  retrieve.py   time-safe BM25 + signals + abstention
  answer.py     answer writer (LLM) + extractive fallback
  actions.py    command -> dry-run actions
  stress.py     leak sweep, paraphrase test, abstention probes (no LLM)
  cli.py        memory | actions | repl
evals/
  memory_probes.jsonl   12 unanswerable probes used by stress.py
run_all.sh      one command for everything   .env.example   requirements.txt
outputs/        train-set outputs from the submitted commit
```