r"""Regenerate every table and inline number in the paper from a results file.

    python eval/make_paper_tables.py eval/results/final_scored.jsonl \
        --out paper/tables [--e2e eval/results/e2e.jsonl]

Writes one `.tex` file per table plus `numbers.tex`, a set of macros
(``\accHosted``, ``\costSaving``, ...) that the prose uses so no percentage is
ever typed by hand into the manuscript. Re-run after any new eval run and the
paper updates itself; a number that moves in the data moves in the text.

`--e2e` is optional. Without it the end-to-end table is emitted with visible
placeholders rather than invented values, so an unfinished run cannot be
mistaken for a finished one at submission time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

CATS = ("intent", "ha", "actuator", "planner", "dynamic")
CAT_LABEL = {
    "intent": "Intent",
    "ha": "HA action",
    "actuator": "Actuator",
    "planner": "Planner",
    "dynamic": "Codegen",
}

# Display names. The harness records provider-qualified ids; the paper wants the
# model, not the transport it happened to be served over.
DISPLAY = {
    "openai:qwen3.5:0.8b": "Qwen3.5 0.8B",
    "openai:qwen3.5:2b": "Qwen3.5 2B",
    "openai:qwen3.5:4b": "Qwen3.5 4B",
    "openai:gemma4:e2b": "Gemma4 E2B",
    "openai:gemma4:e4b": "Gemma4 E4B",
    "openai:gemma4:12b": "Gemma4 12B",
    "openai:gemma4:26b": "Gemma4 26B",
    "anthropic:claude-sonnet-5": "Claude Sonnet 5",
}
# Ascending capacity, hosted model last — the reading order of every table.
ORDER = list(DISPLAY)

PLACEHOLDER = r"\ph"


def esc(text: str) -> str:
    return str(text).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def group_of(case_id: str) -> str:
    parts = case_id.split("-")
    return parts[1] if len(parts) >= 3 and not parts[1].isdigit() else "-"


def strip_fences(text: str) -> str:
    lines = (text or "").strip().splitlines()
    return "\n".join(ln for ln in lines if not ln.strip().startswith("```")).strip()


def extract_json(text: str) -> Any:
    """Same best-effort parse the harness scores with, duplicated here so the
    table script runs without importing the framework (and its dependencies).
    """
    for candidate in (text, strip_fences(text)):
        candidate = (candidate or "").strip()
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except Exception:
            pass
    stripped = strip_fences(text)
    for opener, closer in (("[", "]"), ("{", "}")):
        start, end = stripped.find(opener), stripped.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(stripped[start : end + 1])
            except Exception:
                continue
    return None


class Runs:
    """Indexed view over the records, so every table is one comprehension."""

    def __init__(self, records: list[dict]):
        self.records = records
        self.models = [m for m in ORDER if any(r["model"] == m for r in records)]
        self.models += sorted({r["model"] for r in records} - set(ORDER))

    def sel(self, model=None, category=None) -> list[dict]:
        return [
            r
            for r in self.records
            if (model is None or r["model"] == model)
            and (category is None or r["category"] == category)
        ]

    def acc(self, model, category=None) -> tuple[float, int]:
        rs = self.sel(model, category)
        return (sum(bool(r.get("passed")) for r in rs) / len(rs), len(rs)) if rs else (0.0, 0)

    def cost(self, model, category=None) -> float:
        return sum(r.get("cost_usd") or 0.0 for r in self.sel(model, category))

    def latency(self, model, category=None) -> float | None:
        lats = [r["latency_s"] for r in self.sel(model, category) if r.get("latency_s")]
        return sum(lats) / len(lats) if lats else None

    def throughput(self, model) -> float | None:
        rs = [r for r in self.sel(model) if r.get("latency_s") and r.get("output_tokens")]
        return sum(r["output_tokens"] for r in rs) / sum(r["latency_s"] for r in rs) if rs else None

    def n_cases(self, category) -> int:
        return len({r["case_id"] for r in self.sel(category=category)})


# ── ground truth: which actuator cases must act, which must refuse ───────────
def load_expectations(bench_dir: Path) -> tuple[set[str], set[str]]:
    must_act: set[str] = set()
    must_refuse: set[str] = set()
    for path in sorted(bench_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError:
                continue
            expected = case.get("expected")
            if case.get("category") != "actuator" or not isinstance(expected, list):
                continue
            (must_refuse if expected == [] else must_act).add(case["id"])
    return must_act, must_refuse


def safety_rates(runs: Runs, model: str, must_act: set[str], must_refuse: set[str]):
    """Execution accuracy on actionable cases vs over-actuation on impossible ones.

    Reported as a pair because either one alone is trivially gamed: a model that
    emits `[]` for everything scores a perfect refusal rate, and one that never
    refuses scores well on the actionable half. Only the pair identifies a model
    that is actually usable to drive hardware.
    """
    rows = runs.sel(model, "actuator")
    actionable = [r for r in rows if r["case_id"] in must_act]
    impossible = [r for r in rows if r["case_id"] in must_refuse]
    if not actionable or not impossible:
        return None
    over = sum(
        1
        for r in impossible
        if isinstance(parsed := extract_json(r.get("output") or ""), list) and len(parsed) > 0
    )
    silent = sum(1 for r in actionable if extract_json(r.get("output") or "") == [])
    exec_acc = sum(bool(r.get("passed")) for r in actionable) / len(actionable)
    refuse_acc = 1 - over / len(impossible)
    harmonic = (
        2 * exec_acc * refuse_acc / (exec_acc + refuse_acc) if (exec_acc + refuse_acc) else 0.0
    )
    return {
        "exec": exec_acc,
        "over": over / len(impossible),
        "silent": silent / len(actionable),
        "joint": harmonic,
        "n_act": len(actionable),
        "n_ref": len(impossible),
    }


# ── routing configurations ───────────────────────────────────────────────────
def route_accuracy(runs: Runs, routing: dict[str, str]) -> tuple[float, float]:
    """Accuracy and API spend of a system that sends each call site to one model.

    Accuracy is weighted by the number of benchmark cases at each site, which is
    a property of the benchmark, not of a deployment's call mix — see the paper's
    limitations.
    """
    passed = total = 0.0
    spend = 0.0
    for category, model in routing.items():
        acc, n = runs.acc(model, category)
        passed += acc * n
        total += n
        spend += runs.cost(model, category)
    return passed / total, spend


def best_local(runs: Runs, category: str) -> str:
    local = [m for m in runs.models if not m.startswith("anthropic:")]
    return max(local, key=lambda m: runs.acc(m, category)[0])


# ── table writers ────────────────────────────────────────────────────────────
def tabular(
    body: list[str],
    colspec: str,
    header: str,
    caption: str,
    label: str,
    wide: bool = False,
    here: bool = False,
) -> str:
    """`wide` shrinks a table to the text block — the sub-group tables overflow
    the margin at any readable font size. `here` pins a table in place (`[H]`,
    from the float package) so appendix tables stay under their own heading
    instead of floating above it.
    """
    inner = [
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        header,
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
    ]
    if wide:
        inner = [r"\resizebox{\textwidth}{!}{%", *inner, r"}"]
    return "\n".join(
        [
            r"\begin{table}[H]" if here else r"\begin{table}[t]",
            r"\centering",
            r"\small",
            rf"\caption{{{caption}}}",
            rf"\label{{tab:{label}}}",
            *inner,
            r"\end{table}",
            "",
        ]
    )


def table_main(runs: Runs) -> str:
    body = []
    for model in runs.models:
        cells = []
        for category in CATS:
            acc, _ = runs.acc(model, category)
            cells.append(f"{100 * acc:.1f}")
        overall, _ = runs.acc(model)
        cost = runs.cost(model)
        cost_cell = f"{cost:.2f}" if cost else "0"
        row = f"{DISPLAY.get(model, esc(model))} & " + " & ".join(cells)
        body.append(rf"{row} & \textbf{{{100 * overall:.1f}}} & {cost_cell} \\")
        if model == runs.models[-2]:
            body.append(r"\midrule")
    header = (
        "Model & "
        + " & ".join(CAT_LABEL[c] for c in CATS)
        + r" & All & Cost \\"
        + "\n"
        + " & "
        + " & ".join(f"$n{{=}}{runs.n_cases(c)}$" for c in CATS)
        + rf" & $n{{=}}{len({r['case_id'] for r in runs.records})}$ & (\$) \\"
    )
    return tabular(
        body,
        "l" + "r" * (len(CATS) + 2),
        header,
        "Accuracy (\\%) per call site, one pass over the frozen benchmark at "
        "temperature 0. Local models are served over an OpenAI-compatible endpoint and "
        "cost nothing per call; the hosted column is measured API spend for the whole "
        "280-case sweep.",
        "main",
    )


def table_safety(runs: Runs, must_act: set[str], must_refuse: set[str]) -> str:
    body = []
    n_act = n_ref = 0
    for model in runs.models:
        rates = safety_rates(runs, model, must_act, must_refuse)
        if not rates:
            continue
        n_act, n_ref = rates["n_act"], rates["n_ref"]
        body.append(
            rf"{DISPLAY.get(model, esc(model))} & {100 * rates['exec']:.1f} & "
            rf"{100 * rates['over']:.1f} & {100 * rates['silent']:.1f} & "
            rf"\textbf{{{100 * rates['joint']:.1f}}} \\"
        )
        if model == runs.models[-2]:
            body.append(r"\midrule")
    return tabular(
        body,
        "lrrrr",
        r"Model & Exec.\ acc. & Over-act. & Silent & Joint \\",
        f"Acting versus refusing on the actuator call site ($n{{=}}{n_act}$ actionable, "
        f"$n{{=}}{n_ref}$ impossible). \\emph{{Over-act.}} is the share of impossible requests "
        "that produced at least one service call on real hardware; \\emph{Silent} is the share of "
        "actionable requests answered with an empty action list. \\emph{Joint} is their harmonic "
        "mean. Neither column is meaningful alone: a model that always refuses scores 0 "
        "over-actuation.",
        "safety",
    )


def table_routing(runs: Runs, hosted: str) -> tuple[str, dict]:
    per_site_local = {c: best_local(runs, c) for c in CATS}
    single = max(
        (m for m in runs.models if not m.startswith("anthropic:")),
        key=lambda m: runs.acc(m)[0],
    )
    generative_hosted = dict(per_site_local, planner=hosted, dynamic=hosted)
    configs = [
        ("All hosted", dict.fromkeys(CATS, hosted)),
        ("Hosted for generation only", generative_hosted),
        ("All local, best per site", per_site_local),
        (f"All local, single model ({DISPLAY.get(single, single)})", dict.fromkeys(CATS, single)),
    ]
    body, computed = [], {}
    baseline = None
    for name, routing in configs:
        acc, spend = route_accuracy(runs, routing)
        if baseline is None:
            baseline = spend
        saving = 100 * (1 - spend / baseline) if baseline else 0.0
        per_req = spend / sum(runs.acc(m, c)[1] for c, m in routing.items())
        body.append(rf"{name} & {100 * acc:.1f} & {spend:.2f} & {per_req:.4f} & {saving:.0f} \\")
        computed[name] = {"acc": acc, "spend": spend, "per_req": per_req, "saving": saving}
    return (
        tabular(
            body,
            "lrrrr",
            r"Routing configuration & Acc.\ (\%) & \$ total & \$/call & Saving (\%) \\",
            "Whole-system accuracy and cost under four routing policies over the same 280 cases. "
            "Accuracy is weighted by benchmark case counts, not by a deployment's call mix.",
            "routing",
        ),
        {"configs": computed, "per_site_local": per_site_local, "single": single},
    )


def table_cost_profile(runs: Runs, hosted: str) -> str:
    body = []
    total = runs.cost(hosted)
    for category in CATS:
        rows = runs.sel(hosted, category)
        mean_in = sum(r.get("input_tokens") or 0 for r in rows) / len(rows)
        spend = runs.cost(hosted, category)
        body.append(
            rf"{CAT_LABEL[category]} & {mean_in:,.0f} & {spend:.2f} & "
            rf"{100 * spend / total:.0f} \\".replace(",", r"\,")
        )
    return tabular(
        body,
        "lrrr",
        r"Call site & Mean input tok. & \$ & Share (\%) \\",
        f"Where hosted spend goes ({DISPLAY.get(hosted, hosted)}). The actuator prompt carries the "
        "site's full entity list, so one call site dominates the bill — and it is the site a "
        "mid-sized local model already handles.",
        "costprofile",
        here=True,
    )


def table_latency(runs: Runs) -> str:
    body = []
    for model in runs.models:
        cells = []
        for category in CATS:
            lat = runs.latency(model, category)
            cells.append(f"{lat:.1f}" if lat else "--")
        tps = runs.throughput(model)
        body.append(
            f"{DISPLAY.get(model, esc(model))} & "
            + " & ".join(cells)
            + (rf" & {tps:.0f} \\" if tps else r" & -- \\")
        )
        if model == runs.models[-2]:
            body.append(r"\midrule")
    return tabular(
        body,
        "l" + "r" * (len(CATS) + 1),
        "Model & " + " & ".join(CAT_LABEL[c] for c in CATS) + r" & tok/s \\",
        "Mean wall-clock seconds per call, and aggregate generation throughput. Local latency is "
        "measured on the hardware in \\S\\ref{sec:setup} with no batching; hosted latency includes "
        "network round-trip. Throughput aggregates tokens over time rather than averaging "
        "per-call rates.",
        "latency",
        here=True,
    )


def table_subgroups(
    runs: Runs, category: str, label: str, caption: str, only: set[str] | None = None
) -> str:
    groups = sorted(
        {
            g
            for r in runs.sel(category=category)
            if (g := group_of(r["case_id"])) != "-" and (only is None or g in only)
        }
    )
    if not groups:
        return ""
    counts = {
        g: len({r["case_id"] for r in runs.sel(category=category) if group_of(r["case_id"]) == g})
        for g in groups
    }
    body = []
    for model in runs.models:
        cells = []
        for group in groups:
            rows = [r for r in runs.sel(model, category) if group_of(r["case_id"]) == group]
            cells.append(
                f"{100 * sum(bool(r.get('passed')) for r in rows) / len(rows):.0f}"
                if rows
                else "--"
            )
        body.append(f"{DISPLAY.get(model, esc(model))} & " + " & ".join(cells) + r" \\")
        if model == runs.models[-2]:
            body.append(r"\midrule")
    header = (
        "Model & "
        + " & ".join(esc(g) for g in groups)
        + r" \\"
        + "\n$n$ & "
        + " & ".join(f"{counts[g]}" for g in groups)
        + r" \\"
    )
    return tabular(
        body, "l" + "r" * len(groups), header, caption, label, wide=len(groups) > 8, here=True
    )


def _config_rank(config: str) -> tuple[int, str]:
    """Order routing configurations hosted-to-local, the way the paper reads.

    The label is the operator's, not ours (it only records which LLM_OVERRIDES
    was live), so this is a heuristic with an alphabetical fallback rather than a
    fixed list.
    """
    text = config.lower()
    if "local" in text:
        return (2, text)
    if "hosted" in text and ("gen" in text or "hybrid" in text):
        return (1, text)
    if "hosted" in text:
        return (0, text)
    return (3, text)


def table_e2e(e2e_records: list[dict] | None) -> str:
    """The end-to-end table. With no data it emits placeholders, never numbers.

    Denominators are placeholders too when there are no records: how many cases
    a class contains depends on what the site turns out to own, so writing a
    fixed one here would be a guess dressed as a fact.
    """
    kinds = [
        ("actuate", "Actuate (verified device state change)"),
        ("refuse", "Refuse (no entity anywhere moved)"),
        ("spawn", "Spawn (generated agent registered and ran)"),
        ("nospawn", "No-spawn (correctly built nothing)"),
    ]
    if not e2e_records:
        body = [
            f"{description} & " + " & ".join([rf"{PLACEHOLDER}/{PLACEHOLDER}"] * 3) + r" \\"
            for _, description in kinds
        ]
        header = r"Case class & All hosted & Hosted for gen.\ & All local \\"
        colspec = "lrrr"
        note = ""
    else:
        configs = sorted({r.get("config", "?") for r in e2e_records}, key=_config_rank)
        sites = sorted({r.get("site") or "" for r in e2e_records})
        body = []
        for kind, description in kinds:
            for site in sites:
                cells = []
                for config in configs:
                    rows = [
                        r
                        for r in e2e_records
                        if r.get("kind") == kind
                        and r.get("config") == config
                        and (r.get("site") or "") == site
                        # A case that could not be armed was never asked, so it
                        # belongs in neither the numerator nor the denominator.
                        and r.get("armed") is not False
                    ]
                    cells.append(
                        f"{sum(bool(r.get('passed')) for r in rows)}/{len(rows)}" if rows else "--"
                    )
                label = f"{description}" if len(sites) == 1 else f"{description}, site {esc(site)}"
                body.append(f"{label} & " + " & ".join(cells) + r" \\")
        header = "Case class & " + " & ".join(esc(c) for c in configs) + r" \\"
        colspec = "l" + "r" * len(configs)
        note = (
            f" Run over {len(sites)} installation{'s' if len(sites) > 1 else ''}; "
            "cases whose device the site does not own are not emitted, and a case whose target "
            "could not be armed into a failing state is excluded rather than counted."
        )
    return tabular(
        body,
        colspec,
        header,
        "End-to-end outcomes on the live deployment, verified by side effect rather than by "
        "reading the assistant's reply: an actuate case passes only if the device reaches the "
        "requested state after being armed into a state that does not satisfy the check, a refuse "
        "case only if no actuatable entity in the house moved, a spawn case only if a generated "
        "agent registered --- which requires its code to have executed --- and a no-spawn case "
        "only if the system built nothing for a request that already had an agent, was immediate, "
        "or named devices the site does not have." + note,
        "e2e",
    )


def macros(runs: Runs, routing: dict, hosted: str, must_act, must_refuse, e2e: bool) -> str:
    """Inline numbers the prose cites, so no percentage is hand-typed."""
    cfg = routing["configs"]
    all_hosted = cfg["All hosted"]
    gen_hosted = cfg["Hosted for generation only"]
    local_best = cfg["All local, best per site"]
    single_name = next(k for k in cfg if k.startswith("All local, single"))
    single = cfg[single_name]
    best_local_model = max(
        (m for m in runs.models if not m.startswith("anthropic:")), key=lambda m: runs.acc(m)[0]
    )
    sonnet_safety = safety_rates(runs, hosted, must_act, must_refuse)
    # Worst over-actuator: the abstract quotes it, so it must follow the data.
    rated = [
        (m, safety_rates(runs, m, must_act, must_refuse))
        for m in runs.models
        if safety_rates(runs, m, must_act, must_refuse)
    ]
    worst_model, worst = max(rated, key=lambda pair: pair[1]["over"])
    always_refuse = [m for m, s in rated if s["silent"] > 0.99]
    lines = [
        r"% Generated by eval/make_paper_tables.py -- do not edit by hand.",
        rf"\newcommand{{\nCases}}{{{len({r['case_id'] for r in runs.records})}}}",
        rf"\newcommand{{\nModels}}{{{len(runs.models)}}}",
        rf"\newcommand{{\nRecords}}{{{len(runs.records)}}}",
        rf"\newcommand{{\accHosted}}{{{100 * all_hosted['acc']:.1f}}}",
        rf"\newcommand{{\accLocalBest}}{{{100 * local_best['acc']:.1f}}}",
        rf"\newcommand{{\accGenHosted}}{{{100 * gen_hosted['acc']:.1f}}}",
        rf"\newcommand{{\accSingleLocal}}{{{100 * single['acc']:.1f}}}",
        rf"\newcommand{{\bestLocalModel}}{{{DISPLAY.get(best_local_model, best_local_model)}}}",
        rf"\newcommand{{\costHosted}}{{{all_hosted['spend']:.2f}}}",
        rf"\newcommand{{\costPerCallHosted}}{{{all_hosted['per_req']:.3f}}}",
        rf"\newcommand{{\costGenHosted}}{{{gen_hosted['spend']:.2f}}}",
        rf"\newcommand{{\savingGenHosted}}{{{gen_hosted['saving']:.0f}}}",
        rf"\newcommand{{\gapLocalBest}}{{{100 * (all_hosted['acc'] - local_best['acc']):.1f}}}",
        rf"\newcommand{{\gapGenHosted}}{{{100 * (all_hosted['acc'] - gen_hosted['acc']):.1f}}}",
        rf"\newcommand{{\actuatorCostShare}}"
        rf"{{{100 * runs.cost(hosted, 'actuator') / runs.cost(hosted):.0f}}}",
        rf"\newcommand{{\actuatorMeanTokens}}"
        rf"{{{sum(r.get('input_tokens') or 0 for r in runs.sel(hosted, 'actuator')) / len(runs.sel(hosted, 'actuator')):,.0f}}}".replace(
            ",", r"\,"
        ),
        rf"\newcommand{{\hostedPlannerAcc}}{{{100 * runs.acc(hosted, 'planner')[0]:.0f}}}",
        rf"\newcommand{{\hostedInfeasible}}{{{100 * subgroup_acc(runs, hosted, 'planner', 'infeasible'):.0f}}}",
        rf"\newcommand{{\hostedOverAct}}{{{100 * sonnet_safety['over']:.0f}}}",
        rf"\newcommand{{\maxOverAct}}{{{100 * worst['over']:.1f}}}",
        rf"\newcommand{{\maxOverActModel}}{{{DISPLAY.get(worst_model, worst_model)}}}",
        rf"\newcommand{{\alwaysRefuseModel}}"
        rf"{{{DISPLAY.get(always_refuse[0], always_refuse[0]) if always_refuse else PLACEHOLDER}}}",
        rf"\newcommand{{\eTwoEStatus}}{{{'complete' if e2e else 'PENDING'}}}",
    ]
    # Per-call-site winners, cited in the routing paragraph.
    for category in CATS:
        model = routing["per_site_local"][category]
        acc, _ = runs.acc(model, category)
        lines.append(
            rf"\newcommand{{\best{category.capitalize()}}}"
            rf"{{{DISPLAY.get(model, model)} ({100 * acc:.1f}\%)}}"
        )
    return "\n".join(lines) + "\n"


def subgroup_acc(runs: Runs, model: str, category: str, group: str) -> float:
    rows = [r for r in runs.sel(model, category) if group_of(r["case_id"]) == group]
    return sum(bool(r.get("passed")) for r in rows) / len(rows) if rows else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", help="results.jsonl from harness/evalharness.py")
    ap.add_argument("--e2e", help="records from eval/e2e_run.py (optional)")
    ap.add_argument("--bench", default="benchmark", help="directory holding the benchmark JSONL files")
    ap.add_argument("--out", default="tables", help="where to write the .tex files")
    ap.add_argument("--hosted", default="anthropic:claude-sonnet-5", help="the hosted baseline")
    args = ap.parse_args()

    records = [
        json.loads(line)
        for line in Path(args.results).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    runs = Runs(records)
    must_act, must_refuse = load_expectations(Path(args.bench))
    e2e_records = None
    if args.e2e and Path(args.e2e).exists():
        e2e_records = [
            json.loads(line)
            for line in Path(args.e2e).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    routing_tex, routing = table_routing(runs, args.hosted)

    written = {
        "main.tex": table_main(runs),
        "safety.tex": table_safety(runs, must_act, must_refuse),
        "routing.tex": routing_tex,
        "costprofile.tex": table_cost_profile(runs, args.hosted),
        "latency.tex": table_latency(runs),
        "e2e.tex": table_e2e(e2e_records),
        "sub_planner.tex": table_subgroups(
            runs,
            "planner",
            "subplanner",
            "Planner accuracy by sub-group. \\emph{infeasible} asks for a pipeline the site cannot "
            "support and is scored on refusing to plan one.",
        ),
        # Seventeen sub-groups in one row of columns is unreadable even resized,
        # and the two halves answer different questions anyway: whether a model
        # knows what the site owns, versus how the user named the device.
        "sub_actuator_site.tex": table_subgroups(
            runs,
            "actuator",
            "subactuatorsite",
            "Actuator accuracy by cross-site sub-group. Prefixes \\emph{a\\_}/\\emph{b\\_} are the "
            "two installations; \\emph{control} is a device the site owns, \\emph{flip} one only "
            "the \\emph{other} site owns, \\emph{nearmiss} a plausible name for an absent device, "
            "\\emph{same} a device both own, \\emph{refusal} a device neither owns.",
            only={
                f"{site}_{kind}"
                for kind in ("control", "flip", "nearmiss", "same", "refusal")
                for site in ("a", "b")
            },
        ),
        "sub_actuator_naming.tex": table_subgroups(
            runs,
            "actuator",
            "subactuatornaming",
            "Actuator accuracy by how the device was named: by \\emph{name}, by \\emph{location}, "
            "by both, by an \\emph{attribute} (``the lamp on the small table''), by the "
            "\\emph{switch} it hangs off, or as a \\emph{multi}-device request. \\emph{refusal} "
            "cases name a device the site does not own.",
            only={"name", "location", "name+location", "attribute", "switch", "multi", "refusal"},
        ),
        "sub_dynamic.tex": table_subgroups(
            runs, "dynamic", "subdynamic", "Code generation accuracy by required API shape."
        ),
        "sub_intent.tex": table_subgroups(
            runs, "intent", "subintent", "Intent routing accuracy by sub-group."
        ),
        "sub_ha.tex": table_subgroups(
            runs, "ha", "subha", "Home Assistant action classification accuracy by sub-group."
        ),
        "numbers.tex": macros(runs, routing, args.hosted, must_act, must_refuse, bool(e2e_records)),
    }
    for name, content in written.items():
        if content:
            (out / name).write_text(content, encoding="utf-8")
            print(f"wrote {out / name}")
    if not e2e_records:
        print("\nNOTE: no --e2e records; tab:e2e emitted with \\ph placeholders.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
