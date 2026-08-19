# Runway Response Behavioral-Sanity Review

**Status:** Partial triage completed; behavioral validation remains unresolved.

The June 10 Colab artifact contains 200 generated responses with role labels and scenario identifiers. It does not contain the full prompts, option text, collusion instruction, or honest-agent mandate. Consequently, the saved artifact cannot by itself establish whether a response enacted the assigned collusion instruction.

`scripts/90_runway_reproduction/90_build_behavior_review.py` created a
randomized annotation packet and a separate role key under the ignored
`results/runway_behavior_review/` directory. The script reconstructs the
expected target option label from the deterministic option-swap rule and
identifies only explicit phrases of the form `Option A`, `Option B`, `Option 1`,
or `Option 2`.

| Assigned role | Explicit target mention | Explicit other-option mention | No explicit option mention |
|---|---:|---:|---:|
| Colluder | 20 | 44 | 36 |
| Honest | 2 | 70 | 28 |

These are triage counts, not behavioral success rates. Responses were capped at 64 generated tokens, and an option mention does not show advocacy, deception, or action. Conversely, a response can advocate an option without using the exact `Option X` phrase.

## Required resolution

Before making a behavioral claim about the runway dataset, either:

1. reconstruct the exact upstream prompts, options, and instructions for all 50 scenarios and join them to the saved responses by `scenario_idx`; or
2. rerun generation while saving the complete prompt, displayed options, effective target, role instruction, response, and decoding configuration together.

Future response artifacts must preserve those fields at generation time. Until then, the activation labels should continue to be described as assigned collusion-role labels rather than enacted-deception labels.
