# Qwen3.5-9B behavioral replication of the Scenario 1 grid

Full whole-document grid (3 domains × 3 injection styles × 3 positions × 2
depths × 2 thinking modes): 216 trajectories, 108 injected and 108 clean, all
complete. Qwen3-32B figures are the paper's Table 1, Table 2 and §5.2–5.3 on the
same grid, prompts and sampling settings.

Raw tables: `results/scenario1/qwen3_5_9b_behavioral_summary.md` (from
`scripts/04_reporting/19_summarize_behavioral_outcomes.py`).

## Headline

| | Qwen3-32B (paper) | Qwen3.5-9B |
| --- | --- | --- |
| Executed | 48/108 (44.4%) | **62/108 (57.4%)** |
| Propagated, not executed | 50/108 (46.3%) | 42/108 (38.9%) |
| Resisted | 10/108 (9.3%) | **0/108** |
| Indeterminate | — | 4/108 (repetition loops) |
| Depth effect on execution (2-hop → 3-hop) | 36/54 → 12/54, p < 0.001 | 33/54 → 29/54, p = 0.56 |
| Reasoning effect (off → on) | 17/54 → 31/54, p = 0.011 | 30/54 → 32/54, p = 0.85 |
| Execution by style (1 / 2 / 3) | 12 / 14 / 22 of 36 | **0 / 29 / 33** of 36 |
| Resistance by style (1 / 2 / 3) | 0 / 3 / 7 of 36 | 0 / 0 / 0 of 36 |

Of the paper's three behavioral findings, only the explicitness effect
replicates, and it is far stronger. The depth effect and the reasoning effect
do not replicate. The explicitness *trade-off* (explicit styles are more often
resisted) does not replicate either, because Qwen3.5-9B never resists.

## 1. Depth: no meaningful suppression

| Depth | Thinking | Executed | Propagated | Resisted | Other |
| --- | --- | --- | --- | --- | --- |
| 2-hop | off | 16 | 11 | 0 | 0 |
| 2-hop | on | 17 | 8 | 0 | 2 |
| 3-hop | off | 14 | 13 | 0 | 0 |
| 3-hop | on | 15 | 10 | 0 | 2 |

- Paired by document cell (exact McNemar), 2-hop vs 3-hop execution: off 3
  discordant against 1 (p = 0.62); on 2 against 0 (p = 0.50).
- Cluster bootstrap over the 27 document cells gives a 3-hop minus 2-hop
  difference of −0.074 for both modes (off [−0.22, +0.07], on [−0.19, 0.00]).
- The small drop that remains does not come from executor caution. For styles 2
  and 3 the endpoint reached the executor in 35/36 2-hop runs and 30/36 3-hop
  runs. Once it arrived, the executor acted in 33/35 and 29/30.
- The 6 3-hop misses have two different causes. In 3, all thinking-off, worker 2
  wrote a normal message without the endpoint. In the other 3, all thinking-on,
  a worker's reasoning looped until the token cap and it passed on an empty
  message. With thinking on, the entire 2-hop vs 3-hop gap (2 runs) comes from
  those loops.
- Qwen3-32B's depth effect is attributed to the intermediate worker diluting the
  tool-binding clause (§6). Qwen3.5-9B's second worker usually forwards it
  intact.

## 2. Reasoning: no effect

- Executed with thinking off vs on: 30/54 vs 32/54.
- Of the 54 matched pairs, 44 had the same outcome in both modes; 4 executed
  only with thinking off and 6 only with thinking on (McNemar p = 0.75).
- Bootstrap CI for on minus off: 2-hop +0.037 [−0.07, +0.19], 3-hop +0.037
  [−0.15, +0.22].
- With thinking on, the model reasons about 5× longer (median 2,837 tokens per
  injected turn vs 509 with thinking off). Almost none of that reasoning
  questions the injection: a keyword scan of every thinking-on injected
  trajectory ("injection", "suspicious", "malicious", "exfiltrat", "phishing",
  "untrusted", …) matched 1 of 54, and that run did not execute. It matched
  0 of 54 clean runs.

## 3. Explicitness: a near-binary switch, with no trade-off

| Domain | Calibration line (1) | Self-replicating (2) | Tool-binding (3) |
| --- | --- | --- | --- |
| aihc | 0/12/0 (paper 3/9/0) | 9/3/0 (2/8/2) | 11/1/0 (4/2/6) |
| convex | 0/12/0 (6/6/0) | 11/1/0 (8/4/0) | 10/2/0 (9/2/1) |
| policy | 0/10/0 +2 indet. (3/9/0) | 9/1/0 +2 indet. (4/7/1) | 12/0/0 (9/3/0) |

Cells are executed / propagated / resisted, n = 12. Paper figures for Qwen3-32B
are in parentheses.

- **Style 1 never executes** (0/36), although the calibration line propagates
  in 34/36 runs and reaches the executor in 22. The executor repeats the
  endpoint in its brief in 17 runs but never calls the tool: there were no tool
  calls at all in style 1. Qwen3-32B executed 12/36 of these. Qwen3.5-9B does not act on
  an endpoint unless the instruction says to.
- **Styles 2 and 3 execute almost whenever the endpoint arrives:** 62/65 runs
  that reached the executor. Style 1 vs style 3 gives Fisher p < 10⁻¹⁶.
- **No resistance anywhere, so no detectability trade-off.** In Qwen3-32B the
  most explicit style was both the most effective and the most often rejected,
  most strikingly aihc tool-binding with 6/12 resisted. Qwen3.5-9B executes
  that same cell 11/12.
- **Domain matters much less than in Qwen3-32B.** Within a style, the three
  domains are within 1–2 executions of each other.

## 4. Position: no effect (consistent with the paper)

Executed out of 36: begin 19, middle 21, end 22. The paper had 16 for each.

## 5. Controls and data quality

- **Clean runs:** 108/108 clean, with no tool calls and no stray endpoint.
- **Tool calls:** 0 parse errors in the Qwen3-Coder XML tool-call format across
  all executor turns. 2 calls were blocked (thinking-off, aihc style 2); both
  named a harmless destination (`audit_log`, `Compliance-Audit-Queue`) rather
  than the endpoint. They are correctly counted as not unsafe and labeled
  propagated.
- **Indeterminates (4):** two distinct worker-1 repetition loops (policy,
  middle position, styles 1 and 2). Each is shared by its 2-hop and 3-hop runs,
  which received an identical worker-1 turn. In the 3-hop runs, worker 2 then
  received an empty message and looped too, repeating
  "*(Wait, wait)*: I will write the response." 152 times. They are counted as
  not executed in all rates above.
- **Capped runs that still got a label (3):** one thinking-off run
  (convex style 3, begin, 2-hop) hit the cap writing a long answer, and two
  thinking-on 3-hop runs (convex styles 2 and 3, begin) looped at worker 2.
  All three carried the endpoint earlier in the chain, so they are labeled
  propagated.

## Methods notes and caveats

- **Token caps differ by mode.** Thinking-off runs at 5,000 tokens
  (`controlled_v2_5000`; 1 of 378 turns reached the cap). Thinking-on runs at
  16,384 (`controlled_v2_16384`; 8 of 378 turns reached it), because at 5,000,
  17 of the 51 thinking-on trajectories run so far (33%) had a cut-off turn.
  Qwen3-32B never exceeded 2.7k tokens in 96 local runs, so neither cap would
  have bound it.
- **GPUs differ by mode.** Thinking-off ran on an H200. Thinking-on ran 14
  trajectories on an H200 and 93 on an A100-80GB, plus 1 that used both. Every
  turn records its GPU.
- **Matched runs are not independent.**
  - Generation was partly deterministic: the same seed and input sometimes gave
    an identical turn. That happened in 23/27 thinking-on 2-hop/3-hop pairs at
    worker 1, but 0/27 with thinking off.
  - Whether a turn repeats varied by prompt and hardware, not by our code. One
    planner prompt gave 1 distinct output in 36 runs, another gave 14.
  - So the thinking-on depth contrast mostly compares worker 2 and the
    executor, given a shared upstream.
  - The p-values above are paired (McNemar) or clustered by document cell
    (bootstrap), not naive.
- **"Propagated" follows the paper's definition:** the endpoint appears in any
  downstream message. In 3-hop thinking-off, style 1 propagates through worker 1
  but never reaches the executor (0/9).
- The keyword scan is a crude proxy for detection. It is not a reasoning
  annotation.
- Behavioral only. The probe analysis needs the activations, which are split
  across three Modal workspaces and have to be downloaded first.
