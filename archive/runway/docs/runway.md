# Reproducing the runway baselines

The pre-fellowship runway validated the measurement stack on NARCBench-Core
using Llama 3.1 8B Instruct. These historical values are not Scenario 1
trajectory results.

## Setup

- 50 scenarios and 200 balanced colluder/honest prompts;
- residual-stream extraction across layers 13-20 and 24;
- 4,096-dimensional activation at the last generated token;
- `StandardScaler -> PCA(50) -> logistic regression`;
- five-fold evaluation, leave-one-scenario-out stress testing, Brier score,
  ECE, PCA geometry, direction alignment, and Exemplar Partitioning.

The supported conclusion is narrow: the runway found a modest
in-distribution collusion-role signal that was poorly calibrated, distributed
across layers, and unstable under scenario-held-out evaluation. The labels
describe assigned collusion role, not enacted deception.

| Layer | Pooled AUROC | Brier | ECE | LSO AUROC, mean +/- fold SD |
| ---: | ---: | ---: | ---: | ---: |
| 14 | 0.656 | 0.270 | 0.255 | 0.625 +/- 0.329 |
| 16 | 0.622 | 0.271 | 0.227 | 0.585 +/- 0.349 |
| 20 | 0.641 | 0.278 | 0.256 | 0.595 +/- 0.357 |
| 24 | 0.609 | 0.298 | 0.292 | 0.550 +/- 0.354 |

Compact rerun artifacts live under `results/runway/`.

## Commands

The saved model artifacts are outside Git. Set the artifact root to a directory
containing `02_collusion_probe/week2_collusion_probe_activations.npz` and the
matching response JSON:

```bash
export SPEC_GAP_ARTIFACT_ROOT=/path/to/artifacts
python scripts/90_runway_reproduction/93_run_baselines.py
python scripts/90_runway_reproduction/94_run_lat_baseline.py
```

To create the partial manual review packet:

```bash
python scripts/90_runway_reproduction/90_build_behavior_review.py
```

The packet is written under the ignored `results/runway_behavior_review/`
directory. It is a review aid, not behavioral ground truth, because the old
response artifact did not preserve every full prompt and option.
