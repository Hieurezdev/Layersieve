## A. Retention, failures and causal test (dense-correct prompts)

| schedule | C | prompts | full evidence kept (all) | dense-correct | LS correct | failures | failures that lost evidence | force-keep fixes | placebo fixes | failures with evidence kept | lost evidence but still correct |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ls_l17 | 0.723 | 144 | 144/144 (100%) | 87 | 87 | 0 | 0 | 0/0 | 0/0 | 0 | 0 |
| ls_l14 | 0.644 | 144 | 144/144 (100%) | 87 | 87 | 0 | 0 | 0/0 | 0/0 | 0 | 0 |
| ls_l11 | 0.564 | 144 | 119/144 (83%) | 87 | 60 | 27 | 24 | 24/24 | 0/24 | 3 | 0 |

### A2. Per task

| schedule | task | prompts | full evidence kept | dense-correct | failures | ...lost evidence | force fixes | placebo fixes |
|---|---|---|---|---|---|---|---|---|
| ls_l17 | kv | 24 | 24 | 23 | 0 | 0 | 0 | 0 |
| ls_l17 | vt | 72 | 72 | 19 | 0 | 0 | 0 | 0 |
| ls_l17 | mk | 24 | 24 | 24 | 0 | 0 | 0 | 0 |
| ls_l17 | qa2 | 24 | 24 | 21 | 0 | 0 | 0 | 0 |
| ls_l14 | kv | 24 | 24 | 23 | 0 | 0 | 0 | 0 |
| ls_l14 | vt | 72 | 72 | 19 | 0 | 0 | 0 | 0 |
| ls_l14 | mk | 24 | 24 | 24 | 0 | 0 | 0 | 0 |
| ls_l14 | qa2 | 24 | 24 | 21 | 0 | 0 | 0 | 0 |
| ls_l11 | kv | 24 | 11 | 23 | 12 | 12 | 12 | 0 |
| ls_l11 | vt | 72 | 72 | 19 | 2 | 0 | 0 | 0 |
| ls_l11 | mk | 24 | 12 | 24 | 12 | 12 | 12 | 0 |
| ls_l11 | qa2 | 24 | 24 | 21 | 1 | 0 | 0 | 0 |

## B. Where evidence is lost (every prompt, every lost evidence unit, at its drop layer)

| schedule | lost units (prompts) | first-loss layer: count | pct at cut (median, min-max) | rank at cut (median) | score / best dropped score (median) | lost by task |
|---|---|---|---|---|---|---|
| ls_l17 | 0 | – | – | – | – | – |
| ls_l14 | 0 | – | – | – | – | – |
| ls_l11 | 25 (25) | L12:1, L13:24 | 0.131 (0.002-0.294) | 292 | 0.67 | {'kv': 13, 'mk': 12} |

## C. Are lost evidence units unstable before their cut? (pruned run, alive layers only)


### ls_l11: 25 lost evidence units vs 614663 other dropped units (414353 near the cut, margin >= 0.5)

| feature | lost evidence (median) | evidence kept at same layer (median) | other dropped (median) | AUC lost-ev vs other dropped | AUC vs near-cut dropped |
|---|---|---|---|---|---|
| vol4: std of percentile over the last 4 scored layers (incl. cut) | 0.321 | 0.004 | 0.188 | 0.92 | 0.94 |
| fall: best earlier percentile - percentile at cut | 0.860 | 0.007 | 0.517 | 0.96 | 0.99 |
| peak: best percentile over the scored layers before the cut (dense prefix included) | 0.998 | 0.999 | 0.722 | 1.00 | 1.00 |
| dlast: percentile change from the previous layer (negative = falling) | -0.595 | – | -0.249 | 0.10 | 0.07 |
| head_spread: best-head percentile - median-head percentile at cut | 0.499 | 0.019 | 0.433 | 0.57 | 0.55 |
| head_top: fraction of heads ranking the unit in their top 25% at cut | 0.000 | 0.964 | 0.000 | 0.39 | 0.31 |
| margin: score / best dropped score at cut (near-threshold baseline) | 0.667 | – | 0.622 | 0.55 | 0.37 |
| pct_cut: percentile at cut | 0.131 | 0.991 | 0.188 | 0.36 | 0.18 |

Offline deferral at the first cut: share of the tokens dropped at that layer that must be deferred (ranked by the feature) to also defer the lost evidence unit; lower is better, random = 0.5 on average.

| rank by | median share needed | caught at 5% | at 10% | at 20% |
|---|---|---|---|---|
| vol4 | 0.191 | 0.28 | 0.44 | 0.56 |
| fall | 0.116 | 0.28 | 0.48 | 0.64 |
| peak | 0.011 | 1.00 | 1.00 | 1.00 |
| head_spread | 0.391 | 0.04 | 0.20 | 0.32 |
| head_top | 0.533 | 0.04 | 0.12 | 0.20 |
| margin | 0.573 | 0.04 | 0.04 | 0.08 |
| dlast | 0.263 | 0.20 | 0.32 | 0.44 |

## D. Dense counterfactual: percentile of each lost unit in an unpruned run, at layers after its cut

| schedule | lost units | dense pct at cut layer (median) | mean dense pct over later layers (median) | share of later layers with dense pct >= 0.9 (median) |
|---|---|---|---|---|
| ls_l11 | 25 | 0.321 | 0.998 | 1.00 |

## E. Cost of the measurements (wall clock, shared GPU)

| run | prompts | generate s / prompt (mean) | analysis s / prompt (CPU, mean) | interventions (generate s, total) |
|---|---|---|---|---|
| dense | 144 | 5.49 | – | 0 |
| ls_l17 | 144 | 4.27 | 4.47 | 0 |
| ls_l14 | 144 | 3.85 | 4.05 | 0 |
| ls_l11 | 144 | 3.45 | 3.64 | 174 |
| dense_trace | 25 | 5.68 | 4.00 | 0 |
