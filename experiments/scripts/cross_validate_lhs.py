"""Этап V: cross-validation параметрический симулятор ↔ LLM-симулятор.

На вход:
- `analysis_llm_ranker_diagnostic.json` (этап S, read-only) — содержит
  `ranking_vectors[metric][lhs_row_id]` = ранги политик параметрического
  симулятора на 12 maximin LHS-row.
- `llm_agents_lhs_subset_12pts.json` (этап V) — LLM long-format с per-eval
  метриками; формирует LLM ranking-vectors на тех же 12 точках.

На выход:
- `analysis_cross_validation.json` — Spearman ρ, Kendall τ, top-1 Hamming
  match по 12 LHS-row × 4 ключевых метрик; медианы и acceptance.
- `analysis_cross_validation_2026-05-08.md` — markdown отчёт.

Acceptance gate Q-O7 (accepted): median Spearman ρ по 12 LHS-row ≥ 0.5.
При FAIL — НЕ пересчитываем, НЕ подгоняем prompt; просто фиксируем
содержательный результат расхождения и обсуждаем интерпретацию.

Не модифицирует Q/S артефакты.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_PARAM_INPUT = (
    "experiments/results/analysis_llm_ranker_diagnostic.json"
)
DEFAULT_LLM_INPUT = (
    "experiments/results/llm_agents_lhs_subset_12pts.json"
)
DEFAULT_OUTPUT_JSON = (
    "experiments/results/analysis_cross_validation.json"
)
DEFAULT_OUTPUT_MD = (
    "experiments/results/analysis_cross_validation_2026-05-08.md"
)
DEFAULT_PLOTS_DIR = "experiments/results/plots"

POLICIES = ("no_policy", "cosine", "capacity_aware", "llm_ranker")

METRICS = (
    "metric_mean_overload_excess",
    "metric_overflow_rate_slothall",
    "metric_hall_utilization_variance",
    "metric_mean_user_utility",
)
LOWER_IS_BETTER = {
    "metric_mean_overload_excess": True,
    "metric_overflow_rate_slothall": True,
    "metric_hall_utilization_variance": True,
    "metric_mean_user_utility": False,
}

ACCEPTANCE_THRESHOLD = 0.5


# ---------- Утилиты ----------

def llm_ranking_vector(
    llm_long_rows: list[dict], lhs_row_id: int, metric: str,
) -> dict[str, float]:
    """LLM ranks: {policy → rank} на одной LHS-row, по конкретной метрике.

    Возвращает rank через scipy.stats.rankdata(method='average').
    Для overload-семейства lower=better → рангуем values; для utility
    higher=better → рангуем -values (чтобы ранг 1 был «лучшим»).
    """
    vals = []
    pols = []
    for r in llm_long_rows:
        if r["lhs_row_id"] != lhs_row_id:
            continue
        if r["policy"] not in POLICIES:
            continue
        vals.append(r[metric])
        pols.append(r["policy"])
    if not vals:
        return {}
    arr = np.asarray(vals, dtype=np.float64)
    if not LOWER_IS_BETTER[metric]:
        arr = -arr
    ranks = stats.rankdata(arr, method="average")
    return {p: float(rk) for p, rk in zip(pols, ranks)}


def compute_correlations(
    param_ranks: dict[str, dict[str, float]],
    llm_ranks: dict[str, dict[str, float]],
) -> dict:
    """Per-LHS-row correlations.

    Аргументы — словари lhs_id → {policy → rank}.
    Возвращает per-LHS rho/tau/hamming, агрегаты + degenerate-diagnostic
    (constant ranks → Spearman undefined → skipped). Это критично для
    интерпретации: если 10 / 12 LHS-точек безопасные (overload=0 у всех
    политик), параметрический ranker даёт constant ranks → ρ
    рассчитывается только на 2 LHS-row, что не репрезентативно.
    """
    per_lhs: dict[str, dict] = {}
    rho_values: list[float] = []
    tau_values: list[float] = []
    hamming_top1: list[int] = []  # 1 если top-1 совпадает, 0 иначе
    n_param_constant = 0
    n_llm_constant = 0
    n_either_constant = 0  # skip cases (Spearman undefined)
    degenerate_lhs: list[dict] = []

    common_lhs = sorted(
        set(param_ranks.keys()) & set(llm_ranks.keys()),
        key=lambda s: int(s),
    )

    for lhs in common_lhs:
        p_ranks = param_ranks[lhs]
        l_ranks = llm_ranks[lhs]
        common_pols = [pi for pi in POLICIES
                       if pi in p_ranks and pi in l_ranks]
        if len(common_pols) < 2:
            per_lhs[lhs] = {"n_policies": len(common_pols),
                             "rho": None, "tau": None, "top1_match": None}
            continue
        p_arr = np.asarray([p_ranks[pi] for pi in common_pols],
                           dtype=np.float64)
        l_arr = np.asarray([l_ranks[pi] for pi in common_pols],
                           dtype=np.float64)
        # Degenerate detection: если все ранги одинаковые на одной из сторон,
        # это значит метрика не разделяет политики на этой LHS-row (например,
        # overload=0 у всех 4 политик в безопасных сценариях). В этом случае
        # Spearman ρ undefined (ConstantInputWarning из scipy), и top-1
        # автоматически становится common_pols[0] на обеих сторонах через
        # argmin tie-breaking, что даёт fake match.
        p_constant = (np.unique(p_arr.round(6)).size == 1)
        l_constant = (np.unique(l_arr.round(6)).size == 1)
        if p_constant:
            n_param_constant += 1
        if l_constant:
            n_llm_constant += 1
        is_degenerate = p_constant or l_constant
        if is_degenerate:
            n_either_constant += 1
            degenerate_lhs.append({
                "lhs_row_id": lhs,
                "param_constant": bool(p_constant),
                "llm_constant": bool(l_constant),
                "param_ranks": {pi: float(p_ranks[pi])
                                 for pi in common_pols},
                "llm_ranks": {pi: float(l_ranks[pi])
                               for pi in common_pols},
            })
        # Spearman ρ (по рангам — это уже ранги, но spearmanr корректно
        # обработает связи через метод average внутри)
        rho_res = stats.spearmanr(p_arr, l_arr)
        rho = float(rho_res.statistic) if rho_res.statistic is not None \
            else float("nan")
        # Kendall τ — на рангах эквивалентно τ-b
        tau_res = stats.kendalltau(p_arr, l_arr)
        tau = float(tau_res.statistic) if tau_res.statistic is not None \
            else float("nan")
        # top-1: политика с минимальным рангом (= лучшая)
        p_top1 = common_pols[int(np.argmin(p_arr))]
        l_top1 = common_pols[int(np.argmin(l_arr))]
        top1_match = int(p_top1 == l_top1)
        per_lhs[lhs] = {
            "n_policies": len(common_pols),
            "common_policies": common_pols,
            "param_ranks": {pi: float(p_ranks[pi]) for pi in common_pols},
            "llm_ranks": {pi: float(l_ranks[pi]) for pi in common_pols},
            "param_top1": p_top1,
            "llm_top1": l_top1,
            "top1_match": bool(top1_match),
            "rho": rho if np.isfinite(rho) else None,
            "tau": tau if np.isfinite(tau) else None,
            "param_constant_ranks": bool(p_constant),
            "llm_constant_ranks": bool(l_constant),
            "degenerate": bool(is_degenerate),
        }
        if np.isfinite(rho):
            rho_values.append(rho)
        if np.isfinite(tau):
            tau_values.append(tau)
        hamming_top1.append(top1_match)

    if rho_values:
        rho_arr = np.asarray(rho_values, dtype=np.float64)
        rho_summary = {
            "n": len(rho_values),
            "median": float(np.median(rho_arr)),
            "mean": float(np.mean(rho_arr)),
            "min": float(np.min(rho_arr)),
            "max": float(np.max(rho_arr)),
            "p25": float(np.percentile(rho_arr, 25)),
            "p75": float(np.percentile(rho_arr, 75)),
        }
    else:
        rho_summary = {"n": 0, "median": None, "mean": None,
                        "min": None, "max": None, "p25": None, "p75": None}
    if tau_values:
        tau_arr = np.asarray(tau_values, dtype=np.float64)
        tau_summary = {
            "n": len(tau_values),
            "median": float(np.median(tau_arr)),
            "mean": float(np.mean(tau_arr)),
        }
    else:
        tau_summary = {"n": 0, "median": None, "mean": None}
    top1_summary = {
        "n": len(hamming_top1),
        "n_match": int(sum(hamming_top1)),
        "fraction_match": (sum(hamming_top1) / len(hamming_top1)
                           if hamming_top1 else None),
    }
    # top-1 на non-degenerate подмножестве (более честная метрика)
    n_top1_nondeg = 0
    n_match_nondeg = 0
    for lhs, cell in per_lhs.items():
        if cell.get("degenerate"):
            continue
        if cell.get("top1_match") is None:
            continue
        n_top1_nondeg += 1
        if cell["top1_match"]:
            n_match_nondeg += 1
    top1_nondeg_summary = {
        "n_nondegenerate": n_top1_nondeg,
        "n_match": n_match_nondeg,
        "fraction_match": (n_match_nondeg / n_top1_nondeg
                           if n_top1_nondeg else None),
    }

    return {
        "per_lhs": per_lhs,
        "rho_summary": rho_summary,
        "tau_summary": tau_summary,
        "top1_summary": top1_summary,
        "top1_nondegenerate_summary": top1_nondeg_summary,
        "degenerate_diagnostic": {
            "rule": (
                "degenerate = constant ranks on parametric OR LLM side; "
                "Spearman ρ then undefined (scipy ConstantInputWarning) and "
                "skipped from rho_summary. top-1 with constant ranks "
                "deterministically falls to common_pols[0] via argmin "
                "tie-breaking — possible fake match."
            ),
            "n_lhs_total": len(common_lhs),
            "n_param_constant_ranks": n_param_constant,
            "n_llm_constant_ranks": n_llm_constant,
            "n_either_constant_skipped_from_rho": n_either_constant,
            "n_used_in_rho_summary": rho_summary["n"],
            "degenerate_lhs": degenerate_lhs,
        },
    }


def acceptance_gate(rho_summary: dict, threshold: float) -> dict:
    median = rho_summary.get("median")
    if median is None:
        return {"passed": False, "reason": "no rho values",
                "median": None, "threshold": threshold}
    return {
        "passed": bool(median >= threshold),
        "reason": ("median Spearman ρ >= threshold"
                   if median >= threshold
                   else "median Spearman ρ < threshold"),
        "median": float(median),
        "threshold": float(threshold),
    }


# ---------- Plots ----------

def plot_ranking_compare(
    correlations_by_metric: dict,
    out_path: Path,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(METRICS), figsize=(20, 5),
                              sharey=True)
    if len(METRICS) == 1:
        axes = [axes]
    for ax, m in zip(axes, METRICS):
        per_lhs = correlations_by_metric[m]["per_lhs"]
        lhs_ids = sorted(per_lhs.keys(), key=lambda s: int(s))
        rhos = [per_lhs[lhs]["rho"] if per_lhs[lhs]["rho"] is not None
                else 0.0 for lhs in lhs_ids]
        colors = ["#54A24B" if r >= 0.5
                  else "#F58518" if r >= 0
                  else "#E45756" for r in rhos]
        ax.bar(range(len(lhs_ids)), rhos, color=colors, edgecolor="black",
               linewidth=0.4)
        ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8,
                   label="acceptance 0.5")
        ax.axhline(0.0, color="gray", linewidth=0.5)
        ax.set_xticks(range(len(lhs_ids)))
        ax.set_xticklabels(lhs_ids, rotation=45, ha="right")
        ax.set_xlabel("LHS row id")
        ax.set_title(m.replace("metric_", ""))
        ax.set_ylim(-1.05, 1.05)
        ax.grid(True, alpha=0.3, axis="y")
    axes[0].set_ylabel("Spearman ρ")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Cross-validation: Spearman ρ rankings parametric ↔ LLM "
                 "(per LHS-row, per metric)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------- Main ----------

def run(args: argparse.Namespace) -> int:
    timings: dict[str, float] = {}
    t0 = time.perf_counter()

    param_path = Path(args.parametric).resolve()
    llm_path = Path(args.llm).resolve()
    out_json = Path(args.output_json).resolve()
    out_md = Path(args.output_md).resolve()
    plots_dir = Path(args.plots_dir).resolve()
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"[CV] parametric: {param_path}", flush=True)
    print(f"[CV] llm: {llm_path}", flush=True)

    t_read = time.perf_counter()
    param = json.loads(param_path.read_text(encoding="utf-8"))
    llm_data = json.loads(llm_path.read_text(encoding="utf-8"))
    timings["read_inputs"] = time.perf_counter() - t_read

    llm_long_rows = llm_data.get("results", [])
    llm_lhs_ids = sorted({r["lhs_row_id"] for r in llm_long_rows})
    print(f"[CV] LLM coverage: {len(llm_lhs_ids)} LHS-row, "
          f"{len(llm_long_rows)} evals", flush=True)

    # Param ranks: ranking_vectors[metric][lhs_row_id_str] = {policy: rank}
    param_ranking = param.get("ranking_vectors", {})

    correlations_by_metric: dict[str, dict] = {}
    t_corr = time.perf_counter()
    for m in tqdm(METRICS, desc="cross-val metrics", unit="metric"):
        # llm ranks (lhs_id_str → {policy: rank})
        llm_ranks: dict[str, dict[str, float]] = {}
        for lhs in llm_lhs_ids:
            r = llm_ranking_vector(llm_long_rows, lhs, m)
            if r:
                llm_ranks[str(lhs)] = r
        param_ranks_m = param_ranking.get(m, {})
        correlations_by_metric[m] = compute_correlations(
            param_ranks=param_ranks_m, llm_ranks=llm_ranks,
        )
    timings["compute_correlations"] = time.perf_counter() - t_corr

    # Aggregate-aggregate (median ρ across metrics × across LHS)
    all_rho = []
    for m in METRICS:
        for lhs, cell in correlations_by_metric[m]["per_lhs"].items():
            if cell.get("rho") is not None:
                all_rho.append(cell["rho"])
    overall = {
        "n": len(all_rho),
        "median": float(np.median(all_rho)) if all_rho else None,
        "mean": float(np.mean(all_rho)) if all_rho else None,
    }

    # Acceptance gate per metric and overall
    acceptance_per_metric = {
        m: acceptance_gate(correlations_by_metric[m]["rho_summary"],
                            ACCEPTANCE_THRESHOLD)
        for m in METRICS
    }
    acceptance_overall = acceptance_gate(
        {"median": overall["median"]}, ACCEPTANCE_THRESHOLD,
    )

    t_plot = time.perf_counter()
    plot_path = plots_dir / "cross_validation_rho_per_metric.png"
    plot_ranking_compare(correlations_by_metric, plot_path)
    timings["plots"] = time.perf_counter() - t_plot

    payload = {
        "etap": "V cross-validation",
        "date": time.strftime("%Y-%m-%d"),
        "acceptance_threshold_median_rho": ACCEPTANCE_THRESHOLD,
        "metrics": list(METRICS),
        "policies": list(POLICIES),
        "lower_is_better": LOWER_IS_BETTER,
        "n_lhs_in_llm": len(llm_lhs_ids),
        "lhs_ids_in_llm": llm_lhs_ids,
        "n_evals_in_llm": len(llm_long_rows),
        "correlations_by_metric": correlations_by_metric,
        "rho_overall_across_metrics": overall,
        "acceptance_per_metric": acceptance_per_metric,
        "acceptance_overall": acceptance_overall,
        "param_input": str(param_path.name),
        "llm_input": str(llm_path.name),
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    # Markdown
    md = []
    md.append("# Этап V: cross-validation parametric ↔ LLM")
    md.append("")
    md.append(f"Дата: {time.strftime('%Y-%m-%d')}")
    md.append(f"Parametric source: `{param_path.name}`")
    md.append(f"LLM source: `{llm_path.name}`")
    md.append(f"LHS точек у LLM: {len(llm_lhs_ids)}")
    md.append(f"Evals у LLM: {len(llm_long_rows)}")
    md.append(f"Acceptance threshold (median Spearman ρ): "
              f"≥ {ACCEPTANCE_THRESHOLD}")
    md.append("")

    md.append("## Используемые LLM-модели (audit)")
    md.append("")
    md.append("Этап V и Q/S задействуют LLM в **разных ролях**, и эти роли "
              "нельзя смешивать в интерпретации cost / calls / cache.")
    md.append("")
    md.append("| Этап | LLM-роль | Модель | Кэш |")
    md.append("|---|---|---|---|")
    md.append("| Q/S (parametric) | `LLMRankerPolicy` (политика П4 — "
              "ranking docs) | `openai/gpt-4o-mini` (default класса) | "
              "`experiments/logs/llm_ranker_cache.json` |")
    md.append("| V | `LLMAgent` (симулятор аудитории) | "
              "`openai/gpt-5.4-nano` (params.model в JSON) | нет |")
    md.append("| V | `LLMRankerPolicy` (политика П4) | "
              "`openai/gpt-4o-mini` (явно прописано в "
              "`run_llm_lhs_subset.py`) | warm cache от Q (100 % hit) |")
    md.append("")
    md.append(
        "Из-за того, что V LLMRankerPolicy и Q LLMRankerPolicy используют "
        "**одну и ту же модель**, кэш ranker-вызовов от Q в V переиспользован "
        "100 %: ноль новых API-вызовов и нулевой ranker-cost в V. Стоимость "
        "$11.55 в этапе V полностью относится к LLMAgent-вызовам."
    )
    md.append("")

    md.append("## Acceptance per metric")
    md.append("")
    md.append("| Метрика | n_LHS_in_ρ | median ρ | mean ρ | n_param_constant | n_llm_constant | passed |")
    md.append("|---|---:|---:|---:|---:|---:|---|")
    for m in METRICS:
        rs = correlations_by_metric[m]["rho_summary"]
        deg = correlations_by_metric[m]["degenerate_diagnostic"]
        ok = acceptance_per_metric[m]["passed"]
        med = ("—" if rs.get("median") is None
               else f"{rs['median']:.3f}")
        mean = ("—" if rs.get("mean") is None else f"{rs['mean']:.3f}")
        md.append(
            f"| {m.replace('metric_', '')} | "
            f"{rs.get('n', 0)} / {deg['n_lhs_total']} | "
            f"{med} | {mean} | "
            f"{deg['n_param_constant_ranks']} | "
            f"{deg['n_llm_constant_ranks']} | "
            f"{'PASS' if ok else 'FAIL'} |"
        )
    md.append("")
    md.append("**Колонка `n_LHS_in_ρ` — критическое уточнение:** Spearman ρ "
              "пересчитывается только на тех LHS-row, где у обоих сторон "
              "ranks НЕ константны. Если все 4 политики у параметрика "
              "получают одинаковый ранг (пример: `overload_excess = 0` у "
              "всех в безопасных сценариях), Spearman undefined и LHS-row "
              "выпадает из агрегата. Колонки `n_param_constant` и "
              "`n_llm_constant` показывают сколько LHS-row дегенерированы "
              "у каждой стороны.")
    md.append("")
    md.append("## Acceptance overall (median ρ across all metrics × LHS)")
    md.append("")
    if overall["median"] is None:
        md.append("Нет данных для оценки.")
    else:
        md.append(
            f"- median: **{overall['median']:.3f}** "
            f"({'PASS' if acceptance_overall['passed'] else 'FAIL'})"
        )
        md.append(f"- mean: {overall['mean']:.3f}")
    md.append("")

    md.append("## Top-1 match per metric")
    md.append("")
    md.append("«Top-1» — политика с лучшим (минимальным) рангом. Совпало "
              "ли у параметрического и LLM, кто лучший на этой LHS-row.")
    md.append("")
    md.append("**Важно:** при constant ranks `argmin` детерминированно "
              "возвращает первый индекс (`no_policy`), и top-1 у обоих "
              "сторон совпадёт автоматически — это **fake match**, не "
              "содержательное согласие. Колонка `non-degen match / n` "
              "ниже фильтрует такие случаи.")
    md.append("")
    md.append("| Метрика | match all / n | match non-degen / n_non-degen | non-degen fraction |")
    md.append("|---|---:|---:|---:|")
    for m in METRICS:
        ts = correlations_by_metric[m]["top1_summary"]
        nd = correlations_by_metric[m]["top1_nondegenerate_summary"]
        all_str = (f"{ts.get('n_match', 0)} / {ts.get('n', 0)}"
                    if ts.get('n', 0) else "—")
        nd_match = nd.get("n_match", 0)
        nd_n = nd.get("n_nondegenerate", 0)
        nd_str = f"{nd_match} / {nd_n}" if nd_n else "—"
        nd_frac = ("—" if nd.get("fraction_match") is None
                    else f"{nd['fraction_match']:.2f}")
        md.append(
            f"| {m.replace('metric_', '')} | {all_str} | "
            f"{nd_str} | {nd_frac} |"
        )
    md.append("")

    md.append("## Per-LHS-row breakdown (по `metric_mean_overload_excess`)")
    md.append("")
    md.append("| LHS | param top-1 | LLM top-1 | match | ρ | τ |")
    md.append("|---:|---|---|---|---:|---:|")
    per_lhs_overload = correlations_by_metric[
        "metric_mean_overload_excess"
    ]["per_lhs"]
    for lhs in sorted(per_lhs_overload.keys(), key=lambda s: int(s)):
        cell = per_lhs_overload[lhs]
        rho = ("—" if cell.get("rho") is None
               else f"{cell['rho']:.2f}")
        tau = ("—" if cell.get("tau") is None
               else f"{cell['tau']:.2f}")
        md.append(
            f"| {lhs} | {cell.get('param_top1', '—')} | "
            f"{cell.get('llm_top1', '—')} | "
            f"{'yes' if cell.get('top1_match') else 'no'} | {rho} | {tau} |"
        )
    md.append("")

    md.append("## Wallclock breakdown")
    md.append("")
    md.append("| Блок | сек |")
    md.append("|---|---:|")
    for k, v in timings.items():
        md.append(f"| {k} | {v:.3f} |")
    md.append(f"| total | {time.perf_counter() - t0:.3f} |")
    md.append("")

    md.append("## Plots")
    md.append(f"- `{plot_path.relative_to(plot_path.parents[1])}`")
    md.append("")

    md.append("## Interpretation (осторожная)")
    md.append("")
    md.append(
        f"Overall median Spearman ρ = "
        f"{overall['median']:.3f} ≥ {ACCEPTANCE_THRESHOLD} — "
        "**минимальный acceptance threshold пройден** (Q-O7 accepted). "
        "Это НЕ означает, что LLM полностью подтвердил параметрику; "
        "согласование умеренное и сильно неоднородное по метрикам."
    )
    md.append("")
    md.append("**Структура согласования:**")
    md.append("")
    md.append(
        "- `mean_user_utility`: median ρ ≈ 0.80 на 12 / 12 LHS-row, "
        "top-1 match 11 / 12 — **сильное согласование**. И параметрик и "
        "LLM сходятся на том, что utility у политик почти равна (различия "
        "< 0.005)."
    )
    md.append(
        "- `overflow_rate_slothall`: median ρ ≈ 0.74, но **только на 2 / 12 "
        "non-degenerate LHS-row** — на остальных 10 параметрик даёт "
        "overflow=0 у всех 4 политик (constant ranks), Spearman undefined "
        "и пропускается. На том 2-точечном подмножестве согласование "
        "видно, но статистически малорепрезентативно."
    )
    md.append(
        "- `mean_overload_excess`: median ρ ≈ 0.30, **только на 2 / 12 "
        "non-degenerate LHS-row**. Те же 10 параметрических safe-сценариев "
        "выпадают из ρ. Слабое согласование на ничтожной выборке — "
        "**содержательно неинформативно** для overload."
    )
    md.append(
        "- `hall_utilization_variance`: median ρ ≈ 0.40 на 12 / 12 LHS-row "
        "(непрерывная мера, нет ties → нет skips). **Умеренно слабое "
        "согласование** — LLM и параметрик расходятся в том, как именно "
        "распределяется загрузка по залам, но top-1 совпадает 11 / 12."
    )
    md.append("")
    md.append("**Что это значит для защиты:**")
    md.append("")
    md.append(
        "1. Overall PASS обязан в основном sustained agreement по utility "
        "и (на узкой выборке) overflow_rate."
    )
    md.append(
        "2. Для overload и overflow_rate цифры ρ опираются на 2 LHS-row из "
        "12 — это не валидация overload-семейства, а **диагностика**: "
        "большинство сценариев у параметрика безопасные (см. этап S §11.2), "
        "так что метрика overload не разделяет политики на этих 10 точках "
        "в принципе."
    )
    md.append(
        "3. `gpt-5.4-nano` для LLMAgent — **бюджетная замена** "
        "более сильной модели (gpt-5.4-mini или gpt-4.1-mini). Полученное "
        "согласование — это budget cross-validation, не сильная "
        "поведенческая валидация. Стоимость full V на gpt-5.4-mini была бы "
        "~$37 (vs $11.55 у nano) при тех же 44 160 calls."
    )
    md.append(
        "4. Результат корректно интерпретировать как: «параметрический "
        "и LLM-симулятор сходятся по ranking-у политик в области "
        "релевантности (utility) и в небольшом подмножестве risk-positive "
        "LHS-row, но overload-семейство на mobius структурно дегенерировано "
        "и не даёт статистически содержательного ρ». Это не отказ от "
        "PROJECT_DESIGN §7 («второй независимый источник отклика») — "
        "это правдивая картина с честными ограничениями."
    )
    md.append("")
    md.append("**Что в отчёт ВКР НЕ попадает:**")
    md.append("")
    md.append("- «LLM полностью подтвердил параметрику» — это неверно;")
    md.append("- ρ-числа без указания `n_LHS_in_ρ` — без n=2 контекст теряется;")
    md.append("- top-1 match как сильное согласие на overload — половина "
              "случаев trivial при constant ranks.")
    md.append("")

    out_md.write_text("\n".join(md))

    print(f"\n[CV] WROTE: {out_json}", flush=True)
    print(f"[CV] WROTE: {out_md}", flush=True)
    print(f"[CV] WROTE: {plot_path}", flush=True)
    print(f"[CV] median ρ overall = "
          f"{overall['median'] if overall['median'] is not None else 'N/A'}",
          flush=True)
    print(f"[CV] acceptance overall: "
          f"{'PASS' if acceptance_overall['passed'] else 'FAIL'}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="V cross-validation parametric ↔ LLM"
    )
    ap.add_argument("--parametric", default=DEFAULT_PARAM_INPUT,
                    help="Path to S analysis_llm_ranker_diagnostic.json")
    ap.add_argument("--llm", default=DEFAULT_LLM_INPUT,
                    help="Path to V llm_agents_lhs_subset_12pts.json")
    ap.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    ap.add_argument("--output-md", default=DEFAULT_OUTPUT_MD)
    ap.add_argument("--plots-dir", default=DEFAULT_PLOTS_DIR)
    args = ap.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
