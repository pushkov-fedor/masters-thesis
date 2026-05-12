# LHS-parametric (simplified, plan #35) — mobius_2025_autumn_en

Timestamp: 2026-05-12T21:23:57; wallclock: 9.0s

## Configuration
- n_points: 50 × replicates: 3 × policies: 3
- capacity_multiplier ∈ [0.5, 1.5]
- audience_size = 100 fixed
- popularity_source = cosine_only fixed
- program_variant = 0 fixed
- policies: no_policy, cosine, capacity_aware

## Per-policy mean_overload_excess (across all evals)

| policy | n | mean | median | std | p75 |
|---|---:|---:|---:|---:|---:|
| no_policy | 150 | 0.2248 | 0.0583 | 0.2888 | 0.3850 |
| cosine | 150 | 0.2288 | 0.0654 | 0.2898 | 0.4077 |
| capacity_aware | 150 | 0.1906 | 0.0188 | 0.2697 | 0.3267 |
