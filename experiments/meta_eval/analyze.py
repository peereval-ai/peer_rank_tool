"""Meta-evaluation: do LLM judges agree with humans, and does a PRD-weighted panel beat single judges?

For every human-labeled battle, each judge's preference is averaged over its two display orders
(picks model_a = 1, model_b = 0, tie = 0.5). Methods compared:

  * each single judge
  * Panel (unweighted)   mean preference over judges
  * Panel (PRD-weighted) PRD-weighted mean, weights computed by `peereval score` on the judges'
                         own answers to queries from the same dataset (step 1)

Battle level (as in the PRD paper): accuracy against the human vote (a tie counts 0.5) and
Cohen's kappa. Ranking level (when a human reference ranking is given): each method's verdicts
are turned into a Bradley-Terry ranking of the battle models and compared with the human
ranking (Spearman, Kendall, pairwise order agreement). All numbers have 95% bootstrap
intervals over battles for the battle-level numbers. Rank correlations on resampled battles are
biased low, so at the ranking level only differences between methods (PRD panel vs the best single
judge and vs the unweighted panel, computed on the same resamples) get intervals.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from peereval import normalize, read_jsonl

UNWEIGHTED = "Panel (unweighted)"
PRD = "Panel (PRD-weighted)"


def load_weights(path: str | None, dataset: str) -> dict[str, float]:
    if not path:
        return {}
    data = json.loads(Path(path).read_text())
    if "overall" in data:  # output of `peereval score --json`
        part = data.get("by_dataset", {}).get(dataset) or data["overall"]
        return part["judge_weights"]
    return data.get(dataset, data)  # plain {judge: weight} or {dataset: {judge: weight}}


def preferences(judgments, labels) -> dict[str, dict[str, float]]:
    """[battle][judge] = preference for model_a, averaged over the judge's display orders."""
    acc = defaultdict(lambda: defaultdict(list))
    for j in judgments:
        if j.question_id not in labels:
            continue
        acc[j.question_id][j.judge].append(j.score("model_a"))
    return {b: {judge: sum(v) / len(v) for judge, v in per.items()} for b, per in acc.items()}


def method_prefs(prefs, judges, weights) -> dict[str, dict[str, float]]:
    out = {name: {} for name in [*judges, UNWEIGHTED, PRD]}
    for b, per in prefs.items():
        for judge in judges:
            out[judge][b] = per[judge]
        out[UNWEIGHTED][b] = sum(per[j] for j in judges) / len(judges)
        wsum = sum(weights.get(j, 0.0) for j in judges)
        if wsum > 0:
            out[PRD][b] = sum(weights.get(j, 0.0) * per[j] for j in judges) / wsum
    if not out[PRD]:
        del out[PRD]
    return out


def decision(p: float) -> str:
    return "a" if p > 0.5 else "b" if p < 0.5 else "tie"


def accuracy(pref: dict[str, float], human: dict[str, str], battles: list[str]) -> float:
    s = 0.0
    for b in battles:
        d = decision(pref[b])
        s += 0.5 if d == "tie" else float(d == human[b])
    return s / len(battles)


def kappa(pref: dict[str, float], human: dict[str, str], battles: list[str]) -> float:
    n = len(battles)
    po = sum(decision(pref[b]) == human[b] for b in battles) / n
    cats = ("a", "b", "tie")
    pm = {c: sum(decision(pref[b]) == c for b in battles) / n for c in cats}
    ph = {c: sum(human[b] == c for b in battles) / n for c in cats}
    pe = sum(pm[c] * ph[c] for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def bradley_terry(outcomes: list[tuple[str, str, float]], models: list[str], iters: int = 1000) -> dict[str, float]:
    """outcomes: (model_x, model_y, share won by x)."""
    wins, games = defaultdict(float), defaultdict(float)
    for x, y, p in outcomes:
        wins[x] += p
        wins[y] += 1 - p
        games[tuple(sorted((x, y)))] += 1
    s = {m: 1.0 for m in models}
    for _ in range(iters):
        new = {}
        for i in models:
            den = sum(games[tuple(sorted((i, k)))] / (s[i] + s[k]) for k in models if k != i)
            new[i] = max(wins[i], 1e-9) / den if den else s[i]
        mean = sum(new.values()) / len(new)
        new = {m: v / mean for m, v in new.items()}
        done = max(abs(math.log(new[m] / s[m])) for m in models) < 1e-9
        s = new
        if done:
            break
    return {m: math.log(v) for m, v in s.items()}


def ranks(values: dict[str, float]) -> dict[str, float]:
    order = sorted(values, key=lambda m: -values[m])
    return {m: i for i, m in enumerate(order)}


def spearman(a: dict[str, float], b: dict[str, float]) -> float:
    ms = sorted(a)
    ra, rb = ranks(a), ranks(b)
    n = len(ms)
    d2 = sum((ra[m] - rb[m]) ** 2 for m in ms)
    return 1 - 6 * d2 / (n * (n * n - 1))


def kendall_and_agreement(a: dict[str, float], b: dict[str, float]) -> tuple[float, float]:
    ms = sorted(a)
    conc = disc = 0
    for i in range(len(ms)):
        for k in range(i + 1, len(ms)):
            s = (a[ms[i]] - a[ms[k]]) * (b[ms[i]] - b[ms[k]])
            conc += s > 0
            disc += s < 0
    pairs = len(ms) * (len(ms) - 1) / 2
    return (conc - disc) / pairs, conc / pairs


def ranking_metrics(pref, battles, labels, human_bt, models):
    outcomes = [(labels[b]["model_a"], labels[b]["model_b"], pref[b]) for b in battles]
    bt = bradley_terry(outcomes, models)
    tau, agree = kendall_and_agreement(bt, human_bt)
    return {"spearman": spearman(bt, human_bt), "kendall": tau, "pair_agreement": agree}


def ci(values: list[float]) -> list[float]:
    """95% percentile bootstrap interval."""
    v = sorted(x for x in values if x == x)
    if not v:
        return [float("nan"), float("nan")]
    return [v[int(0.025 * (len(v) - 1))], v[int(round(0.975 * (len(v) - 1)))]]


def analyze_dataset(name, judgments, labels, weights, human_rank, n_boot, seed):
    prefs = preferences(judgments, labels)
    judges = sorted({j for per in prefs.values() for j in per})
    battles = sorted(b for b, per in prefs.items() if all(j in per for j in judges))
    human = {b: "a" if labels[b]["human_winner"] == "model_a" else "b" for b in battles}
    methods = method_prefs({b: prefs[b] for b in battles}, judges, weights)
    singles = [j for j in judges]

    def battle_level(bs):
        return {m: {"accuracy": accuracy(p, human, bs), "kappa": kappa(p, human, bs)} for m, p in methods.items()}

    point = battle_level(battles)
    best_single = max(singles, key=lambda j: point[j]["accuracy"])

    rank_point = rank_models = human_bt = None
    if human_rank:
        rank_models = [m for m in human_rank["ranking"]]
        human_bt = {m: human_rank["bradley_terry"][m] for m in rank_models}
        in_scope = [b for b in battles if labels[b]["model_a"] in human_bt and labels[b]["model_b"] in human_bt]
        human_pref = {b: 1.0 if human[b] == "a" else 0.0 for b in in_scope}
        rank_methods = dict(methods, **{"Human votes on these battles": human_pref})
        rank_point = {m: ranking_metrics(p, in_scope, labels, human_bt, rank_models) for m, p in rank_methods.items()}
        best_rank_single = max(singles, key=lambda j: rank_point[j]["spearman"])

    rng = random.Random(seed)
    boot = defaultdict(list)
    for _ in range(n_boot):
        bs = [rng.choice(battles) for _ in battles]
        bl = battle_level(bs)
        for m, v in bl.items():
            boot[(m, "accuracy")].append(v["accuracy"])
            boot[(m, "kappa")].append(v["kappa"])
        if PRD in bl:
            boot[("diff", "prd_minus_best_single")].append(bl[PRD]["accuracy"] - bl[best_single]["accuracy"])
            boot[("diff", "prd_minus_unweighted")].append(bl[PRD]["accuracy"] - bl[UNWEIGHTED]["accuracy"])
        if rank_point:
            sc = [b for b in bs if labels[b]["model_a"] in human_bt and labels[b]["model_b"] in human_bt]
            rm = {m: ranking_metrics(rank_methods[m], sc, labels, human_bt, rank_models)
                  for m in (PRD, UNWEIGHTED, best_rank_single) if m in rank_methods}
            if PRD in rm:
                for k in rm[PRD]:
                    boot[("rank_diff_best", k)].append(rm[PRD][k] - rm[best_rank_single][k])
                    boot[("rank_diff_unweighted", k)].append(rm[PRD][k] - rm[UNWEIGHTED][k])

    result = {"dataset": name, "battles": len(battles), "judges": judges, "weights": {j: weights.get(j) for j in judges},
              "best_single_judge": best_single,
              "battle_level": {m: {k: {"value": v, "ci": ci(boot[(m, k)])} for k, v in d.items()} for m, d in point.items()}}
    if PRD in point:
        d1 = point[PRD]["accuracy"] - point[best_single]["accuracy"]
        d2 = point[PRD]["accuracy"] - point[UNWEIGHTED]["accuracy"]
        result["prd_vs_best_single"] = {"diff": d1, "ci": ci(boot[("diff", "prd_minus_best_single")])}
        result["prd_vs_unweighted"] = {"diff": d2, "ci": ci(boot[("diff", "prd_minus_unweighted")])}
    if rank_point:
        result["ranking_level"] = rank_point
        result["ranking_models"] = rank_models
        result["best_ranking_single_judge"] = best_rank_single
        if PRD in rank_point:
            result["ranking_diffs"] = {
                label: {k: {"diff": rank_point[PRD][k] - rank_point[other][k], "ci": ci(boot[(key, k)])}
                        for k in rank_point[PRD]}
                for label, key, other in (("prd_minus_best_single", "rank_diff_best", best_rank_single),
                                          ("prd_minus_unweighted", "rank_diff_unweighted", UNWEIGHTED))}
    return result


def fmt(x):
    return f"{x['value']:.3f} ({x['ci'][0]:.3f}–{x['ci'][1]:.3f})"


def to_markdown(results) -> str:
    out = ["# Meta-evaluation results\n"]
    for r in results:
        out.append(f"## {r['dataset']} ({r['battles']} battles, {len(r['judges'])} judges)\n")
        out.append("| Method | Accuracy vs human (95% CI) | Cohen's kappa (95% CI) | PRD weight |")
        out.append("|---|---|---|---:|")
        order = sorted(r["battle_level"], key=lambda m: -r["battle_level"][m]["accuracy"]["value"])
        for m in order:
            v = r["battle_level"][m]
            w = r["weights"].get(m)
            out.append(f"| {m} | {fmt(v['accuracy'])} | {fmt(v['kappa'])} | {'' if w is None else f'{w:.3f}'} |")
        if "prd_vs_best_single" in r:
            d1, d2 = r["prd_vs_best_single"], r["prd_vs_unweighted"]
            out.append(f"\nPRD panel minus best single judge ({r['best_single_judge']}): "
                       f"{d1['diff']:+.3f} (95% CI {d1['ci'][0]:+.3f} to {d1['ci'][1]:+.3f})  ")
            out.append(f"PRD panel minus unweighted panel: {d2['diff']:+.3f} "
                       f"(95% CI {d2['ci'][0]:+.3f} to {d2['ci'][1]:+.3f})")
            out.append("\n(The best single judge is picked on the same data, so its accuracy is optimistic.)\n")
        if "ranking_level" in r:
            out.append(f"\n### Ranking of {len(r['ranking_models'])} battle models vs the human ranking\n")
            out.append("| Method | Spearman | Kendall | Pairwise order agreement |")
            out.append("|---|---:|---:|---:|")
            for m, v in sorted(r["ranking_level"].items(), key=lambda kv: -kv[1]["spearman"]):
                out.append(f"| {m} | {v['spearman']:.3f} | {v['kendall']:.3f} | {v['pair_agreement']:.3f} |")
            if "ranking_diffs" in r:
                names = {"prd_minus_best_single": f"PRD panel minus best single judge ({r['best_ranking_single_judge']})",
                         "prd_minus_unweighted": "PRD panel minus unweighted panel"}
                out.append("\nDifferences with 95% bootstrap intervals (both methods on the same resampled battles):\n")
                out.append("| Comparison | Spearman | Kendall | Pairwise order agreement |")
                out.append("|---|---|---|---|")
                for key, d in r["ranking_diffs"].items():
                    cells = [f"{d[k]['diff']:+.3f} ({d[k]['ci'][0]:+.3f} to {d[k]['ci'][1]:+.3f})"
                             for k in ("spearman", "kendall", "pair_agreement")]
                    out.append(f"| {names[key]} | " + " | ".join(cells) + " |")
            out.append("\n\"Human votes on these battles\" ranks the models from the sampled human votes alone; "
                       "it shows how close any method can get with this many battles.\n")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", nargs="+", required=True, help="data/<name>_labels.jsonl files")
    ap.add_argument("--judgments", nargs="+", required=True, help="judgments of the human-labeled battles (step 2)")
    ap.add_argument("--weights", help="`peereval score --json` output of step 1 (or a {judge: weight} JSON)")
    ap.add_argument("--human-ranking", nargs="*", default=[], help="data/<name>_human_ranking.json files")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    labels_by_ds = defaultdict(dict)
    for row in read_jsonl(args.labels):
        labels_by_ds[row["dataset"]][str(row["question_id"])] = row
    rankings = {}
    for path in args.human_ranking:
        name = Path(path).name.replace("_human_ranking.json", "")
        rankings[name] = json.loads(Path(path).read_text())
    judgments = normalize(read_jsonl(args.judgments))

    results = []
    for ds, labels in sorted(labels_by_ds.items()):
        js = [j for j in judgments if j.dataset == ds]
        if not js:
            print(f"{ds}: no judgments found, skipped")
            continue
        r = analyze_dataset(ds, js, labels, load_weights(args.weights, ds), rankings.get(ds), args.bootstrap, args.seed)
        results.append(r)
        print(f"{ds}: {r['battles']} battles judged by all {len(r['judges'])} judges")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta_eval.json").write_text(json.dumps(results, indent=2))
    md = to_markdown(results)
    (out / "meta_eval.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
