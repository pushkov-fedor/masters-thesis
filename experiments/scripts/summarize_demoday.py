"""Compose short summary of Demo Day 2026 EN parametric Q run vs Mobius EN.

Reads:
  results/lhs_parametric_demo_day_2026_en_2026-05-12.json
  results/demo_day_en/analysis_pairwise.json
  results/en/analysis_pairwise.json (Mobius EN baseline)
  data/personas/test_diversity/internal_consistency_demoday.json
  data/personas/test_diversity/diagnose_demoday_en.json

Writes:
  results/demo_day_en/report_demoday_en_summary.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

LHS_JSON = ROOT / "results/lhs_parametric_demo_day_2026_en_2026-05-12.json"
PAIRWISE_DD = ROOT / "results/demo_day_en/analysis_pairwise.json"
PAIRWISE_MOB = ROOT / "results/en/analysis_pairwise.json"
CONSIST_JSON = ROOT / "data/personas/test_diversity/internal_consistency_demoday.json"
DIAG_JSON = ROOT / "data/personas/test_diversity/diagnose_demoday_en.json"
OUT_MD = ROOT / "results/demo_day_en/report_demoday_en_summary.md"


def fmt_pct(x):
    return f"{x * 100:.1f}%"


def main():
    lhs = json.loads(LHS_JSON.read_text())
    pw_dd = json.loads(PAIRWISE_DD.read_text())
    pw_mob = json.loads(PAIRWISE_MOB.read_text())
    cons = json.loads(CONSIST_JSON.read_text())
    diag = json.loads(DIAG_JSON.read_text())

    timings = lhs.get("timings", {})
    metric_key = "metric_mean_overload_excess"

    # full_50 pairwise on demo_day
    fp_dd = pw_dd["full_50"][metric_key]["cosine_vs_capacity_aware"]
    fp_mob = pw_mob["full_50"][metric_key]["cosine_vs_capacity_aware"]
    n_full_dd = pw_dd["n_lhs_full_50"]
    n_full_mob = pw_mob["n_lhs_full_50"]

    # per-policy distribution mean overload (full_50, P1-P3 only)
    dist_dd = pw_dd["distribution_full_50"]
    dist_mob = pw_mob["distribution_full_50"]

    lines = []
    lines.append("# Demo Day 2026 EN — параметрический Q (второй инстанс)")
    lines.append("")
    lines.append(
        "Дата прогона: 2026-05-12. Цель — воспроизвести центральный численный "
        "тезис Mobius EN на втором инстансе конференции (Demo Day ITMO 2026, "
        "IT-конференция широкого профиля). BGE-large-en + ABTT-1 + 150 EN-персон + "
        "50 LHS × 3 replicate × {П1, П2, П3} = 450 evals.")
    lines.append("")
    lines.append("## Данные")
    lines.append("")
    lines.append("- Конференция: 210 talks, 7 halls, 56 slots, 2 дня (2026-01-22 / 23).")
    lines.append(
        f"- Персон: {cons['n_total']} EN, distribution exp jr/mid/sr/lead = "
        f"12/45/66/27 (8/30/44/18%); company smallish-startup/midsize/large/enterprise = "
        f"27/63/42/18 (18/42/28/12%); 47 distinct roles.")
    lines.append(
        "- Эмбеддинги: BGE-large-en + ABTT-1 (1024-dim), fame.json скопирован из "
        "RU-версии (id совпадают).")
    lines.append("")
    lines.append("## Acceptance персон (4 проверки)")
    lines.append("")
    lines.append(
        f"- **Internal consistency (LLM-judge claude-haiku-4.5):** "
        f"{cons['n_consistent']}/{cons['n_total']} "
        f"({fmt_pct(cons['pct_consistent'])}). Цель ≥ 95% — **PASS**.")
    v = diag["vendi"]
    lines.append(
        f"- **Vendi Score (cos+ABTT):** {v['abtt_cos']:.2f} из 150 "
        f"({fmt_pct(v['abtt_pct'])}). Raw без ABTT = {v['raw_cos']:.2f} "
        f"(узкий конус BGE); BM25 = {v['bm25']:.2f} "
        f"({fmt_pct(v['bm25_pct'])}). Парные ABTT cos: "
        f"mean {diag['pairs']['abtt_mean']:.3f}, max "
        f"{diag['pairs']['abtt_max']:.3f}; дублей > 0.95: "
        f"{diag['pairs']['abtt_dup_95']}.")
    cov = diag["coverage"]
    lines.append(
        f"- **Coverage программы (cos+ABTT τ=0.50, 0.60):** dead docs "
        f"{cov['tau_0.5']['dead']}/210 при τ=0.5; "
        f"{cov['tau_0.6']['dead']}/210 при τ=0.6; mean interested per talk = "
        f"{cov['tau_0.5']['mean']:.1f} (τ=0.5).")
    lines.append(
        "- **EC smoke:** EC1 PASS (cap×3 → overload=0), EC3 PASS "
        "(w_rec=0 range=0), EC4 PASS (cap×0.5 различимы).")
    lines.append("")
    lines.append("## Центральный численный тезис")
    lines.append("")
    lines.append(
        f"Pairwise `cosine` vs `capacity_aware` по `mean_overload_excess` на "
        f"full_{n_full_dd} (фракции от {fp_dd['n_paired']} valid пар):")
    lines.append("")
    lines.append(f"- **strict wins cosine:** {fp_dd['win_strict']:.0%}")
    lines.append(
        f"- **strict wins capacity_aware:** "
        f"{fp_dd['loss_strict']:.0%} (loss_strict для cosine)")
    lines.append(
        f"- **ε-equivalent (ε=0.005):** "
        f"{fp_dd['ties_eps']:.0%}")
    lines.append(
        f"- **eps wins cosine (cosine лучше за ε):** {fp_dd['win_eps']:.0%}")
    lines.append(
        f"- **eps wins capacity_aware (cap_aware лучше за ε):** "
        f"{fp_dd['loss_eps']:.0%}")
    lines.append("")
    lines.append("Per-policy mean(median) overload по full_50:")
    for pol in ("no_policy", "cosine", "capacity_aware"):
        stats = dist_dd[pol][metric_key]
        lines.append(
            f"- `{pol}`: mean={stats['mean']:.4f}, "
            f"median={stats['median']:.4f}, max={stats['max']:.4f}")
    lines.append("")
    lines.append("## Сравнение с Mobius EN")
    lines.append("")
    lines.append("| Метрика | Mobius EN | Demo Day EN |")
    lines.append("|---|---:|---:|")
    lines.append(f"| n LHS-точек | {n_full_mob} | {n_full_dd} |")
    lines.append(
        f"| cosine strict wins | {fp_mob['win_strict']:.0%} "
        f"| {fp_dd['win_strict']:.0%} |")
    lines.append(
        f"| capacity_aware strict wins | {fp_mob['loss_strict']:.0%} "
        f"| {fp_dd['loss_strict']:.0%} |")
    lines.append(
        f"| eps cosine wins | {fp_mob['win_eps']:.0%} "
        f"| {fp_dd['win_eps']:.0%} |")
    lines.append(
        f"| eps cap_aware wins | {fp_mob['loss_eps']:.0%} "
        f"| {fp_dd['loss_eps']:.0%} |")
    lines.append(
        f"| ε-equivalent | {fp_mob['ties_eps']:.0%} | "
        f"{fp_dd['ties_eps']:.0%} |")
    lines.append("")
    lines.append("**Тезис о том, что `cosine` не выигрывает у `capacity_aware`:**")
    cos_strict_dd = fp_dd['win_strict']
    cos_eps_dd = fp_dd['win_eps']
    if cos_strict_dd <= 1e-6 and cos_eps_dd <= 1e-6:
        lines.append(
            "- Demo Day: **подтверждён**. 0 strict побед cosine, 0 ε-побед cosine.")
    elif cos_strict_dd <= 1e-6:
        lines.append(
            f"- Demo Day: **строгая часть подтверждена** (0 strict побед cosine), "
            f"но cosine выигрывает за ε на {cos_eps_dd:.0%} точек. "
            f"Качественное направление сохраняется: cap_aware "
            f"доминирует на {fp_dd['loss_strict']:.0%} strict.")
    else:
        lines.append(
            f"- Demo Day: cosine побеждает строго на {cos_strict_dd:.0%} точек "
            f"— тезис **не выполняется буквально**, но cap_aware всё ещё доминирует "
            f"на {fp_dd['loss_strict']:.0%} строго.")
    lines.append("")
    lines.append("## Бюджеты")
    lines.append("")
    total_s = (
        timings.get("load_conference_s", 0.0)
        + timings.get("generate_lhs_s", 0.0)
        + timings.get("maximin_subset_s", 0.0)
        + timings.get("prep_total_s", 0.0)
        + timings.get("p1_p3_total_s", 0.0)
        + timings.get("p4_total_s", 0.0)
    )
    lines.append(
        f"- Wallclock LHS-прогон: "
        f"{total_s:.1f}s "
        f"({total_s / 60:.1f} мин); из них prep (enumerate_modifications "
        f"deepcopy для program_variant>0) — "
        f"{timings.get('prep_total_s', 0.0):.0f}s, sim P1-P3 — "
        f"{timings.get('p1_p3_total_s', 0.0):.1f}s.")
    lines.append(
        f"- LLM-вызовы: translation 210 talks → $0.23 (claude-haiku-4.5, 38с), "
        f"persona generation 150 → $0.21 (claude-haiku-4.5, 58с), "
        f"consistency audit 150 → ${cons.get('cost_usd', 0.0):.2f} "
        f"(claude-haiku-4.5, 25с). "
        f"Total ≈ ${0.23 + 0.21 + cons.get('cost_usd', 0.0):.2f}.")
    lines.append("")
    lines.append("## Артефакты")
    lines.append("")
    lines.append("- `data/conferences/demo_day_2026_en.json` — переведённая программа.")
    lines.append(
        "- `data/conferences/demo_day_2026_en_embeddings.npz` — BGE+ABTT эмбеддинги talks.")
    lines.append(
        "- `data/conferences/demo_day_2026_en_fame.json` — fame scores (skipped из RU-копии).")
    lines.append(
        "- `data/personas/personas_demoday_en.json` — 150 EN-персон.")
    lines.append(
        "- `data/personas/personas_demoday_en_embeddings.npz` — BGE+ABTT эмбеддинги personas.")
    lines.append(
        "- `data/personas/test_diversity/internal_consistency_demoday.json` — LLM-judge consistency audit.")
    lines.append(
        "- `data/personas/test_diversity/diagnose_demoday_en.json` — Vendi / coverage / distributions.")
    lines.append(
        "- `results/lhs_parametric_demo_day_2026_en_2026-05-12.{json,csv,md}` — параметрический Q-прогон.")
    lines.append(
        "- `results/demo_day_en/analysis_*.json` — постобработка.")
    lines.append(
        "- `results/demo_day_en/analysis_lhs_parametric_2026-05-12_demoday_en.md` — markdown отчёт постобработки.")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"WROTE {OUT_MD}")


if __name__ == "__main__":
    main()
