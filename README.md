# slm-callsite-eval

Benchmark, evaluation harness and scored records for the paper

> **Not Every Call Needs a Frontier Model: Per-Call-Site Evaluation of Small
> Language Models in a Deployed Agentic Home-Automation System**
> Panagiotis Kasnesis, Christos Chatzigeorgiou, Lazaros Toumanidis,
> Amalia Contiero Syropoulou.
> 1st Workshop on SLMs for Agentic Systems (SLM-Agents), NeurIPS 2026, Paris.
> [OpenReview](https://openreview.net/forum?id=H5z1pZCS4E)

The system under evaluation is [Wactorz](https://github.com/waldiez/wactorz),
an open-source multi-agent home-automation framework. All experiments use the
production prompts of **Wactorz v0.5.3**, and the live end-to-end run used a
v0.5.3-based deployment. Later releases have changed these prompts, so pin
v0.5.3 when re-running (see `requirements.txt`).

## Layout

    benchmark/   280 cases across five call sites
                   intent_ha_hard.jsonl     36 intent + 28 HA action
                   actuator.jsonl,          116 actuator cases, two installations
                   actuator_curated.jsonl,    (site B and cross-site cases ship
                   actuator_b_keys.jsonl,      as keys only; see Privacy)
                   crosssite_keys.jsonl
                   planner_grounded.jsonl   50 planner
                   dynamic_strict.jsonl     50 codegen
    harness/     evalharness.py            runs a model over a case file
                 make_bench_*.py           generators the cases came from
                 analyze_results.py        scoring
                 make_paper_tables.py      every table in the paper
    results/     final_scored.jsonl        2520 records: 9 models x 280 cases,
                                           one pass, temperature 0
                 e2e_user_eval.jsonl       46 live cases x 3 routing configs,
                                           judged by a person

## Reproducing the paper's tables

Every number in the paper is generated, none typed by hand. This needs only
Python 3.10+ and no third-party packages:

    python3 harness/make_paper_tables.py results/final_scored.jsonl \
      --e2e results/e2e_user_eval.jsonl --bench benchmark --out tables

This writes the `.tex` tables and `numbers.tex` (the inline macros) to
`tables/`.

## Re-running models

Re-running needs the framework, because the harness imports the production
prompts and the LLM provider factory from it:

    python3 -m venv .venv && . .venv/bin/activate
    pip install -r requirements.txt
    python harness/evalharness.py --models "ollama:qwen3:4b" \
      --prompts benchmark/intent_ha_hard.jsonl --temperature 0 --out results/

See `python harness/evalharness.py --help` for all options. Provider
credentials and endpoints are configured as in Wactorz v0.5.3.

## Record schema

One JSON object per line: `model`, `case_id`, `category`, `rep`, `passed`,
`output`, `input_tokens`, `output_tokens`, `latency_s`, `cost_usd`, `error`.
`category` is the call site, and `case_id` encodes the sub-group used in the
appendix breakdowns.

## Answer keys

Three `intent` keys and the planner entity block were revised after inspecting
model outputs, for the reasons given in the paper's Limitations section. Both
key sets are included so the revision can be checked. The revisions moved the
hosted model +0.08 and three local models -0.03, against the direction of the
paper's conclusion.

## Prompts

The five prompts are reproduced verbatim in Appendix A of the paper. The first
three are imported unchanged from Wactorz v0.5.3. The planner and codegen
prompts are condensed variants that keep the output contract scoring depends on.

## Privacy

The benchmark is grounded in two installations in an occupied home, so some
material is withheld or redacted:

- Device registries are not shipped. They carry network and location
  identifiers.
- `actuator_b_keys.jsonl` (14 cases, site B) and `crosssite_keys.jsonl`
  (32 cases) ship as keys only: case id and answer key, with the prompt
  removed, because their device payloads describe a resident's phone and the
  building's network. Every scoring table reproduces from the keys alone,
  since scoring compares the emitted action set against the key and never reads
  the prompt. What the keys do not permit is re-running a model on those 46
  prompts. The other 70 actuator cases and every intent, HA-action, planner
  and codegen case ship in full.
- Private addresses in recorded prompts and outputs are replaced with the
  RFC 5737 documentation range. Recorded `input_tokens` describe the original
  text, so re-tokenising a redacted prompt may differ by a token or two.

## License

Apache-2.0, matching Wactorz.

## Acknowledgments

Research supported by the NVIDIA Academic Grant Program using two NVIDIA DGX
Spark systems.
