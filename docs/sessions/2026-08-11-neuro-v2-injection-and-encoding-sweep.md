# What this repo does, and what we did to it tonight

A walkthrough of the SPEC-GAP pipeline for anyone picking this up cold,
followed by a log of the 2026-08-11 session: standing up a new neuro
injection variant end to end, and a detour into fixing a Windows encoding
bug that turned out to run through the whole codebase.

| | |
| --- | --- |
| Repo | spec-gap |
| Domain worked | neuro |
| Model | `Qwen/Qwen3-32B` rev `9216db5` |
| Spend | $0.00 |

**Status at end of session:** injection built and verified (pixel-identical
render) &middot; local pipeline S00&ndash;S06 passing &middot; Modal H200
blocked on billing &middot; encoding sweep across 49 files.

## What SPEC-GAP is for

SPEC-GAP studies the gap between what a language-model agent *appears* to do
and what its internal activations show it is doing. The testbed is an
indirect prompt injection: a hostile instruction hidden inside one retrieved
document, planted so that a human skimming the file would not notice it.
That document is fed into a small multi-agent pipeline &mdash; a **planner**,
one or two **workers**, and an **executor** &mdash; and the question is
twofold: does the agent chain *behave* as if it read the hidden instruction,
and do the model's **residual-stream activations** encode that exposure even
when the visible output looks clean?

Every domain package (neuro, telecom, policy, petro, fin, &hellip;) ships as
a matched pair: three genuine clean documents, and one injected twin of a
single document in that set. The injected condition swaps only that one
carrier document in; everything else about the task, retrieval, and prompts
stays identical. That's what makes the clean/injected contrast usable as a
controlled experiment rather than an anecdote.

> **Where the evidence currently stands.** The frozen 2026-08-06 cohort
> covers 9 domains, 72 trajectories, 252 model turns, 630 activation
> checkpoints across all 64 residual layers of Qwen3-32B. Worker 1, thinking
> off, layer 40 reaches mean held-out-domain AUROC **0.889** &mdash; but
> that number is for separating clean vs. injected *construction*, not for
> detecting a successful behavioral compromise. All 36 injected trajectories
> in that cohort resisted under the automatic endpoint rule; the two-person
> human behavioral review is still open and the repo treats it as blank
> until reviewers lock it &mdash; nothing here gets retroactively upgraded
> to "definitive."

## The pipeline, S00&ndash;S23

One canonical run order, spanning three very different cost tiers: free and
local, remote-but-still-free, and paid GPU. Tonight's work lived entirely in
the first tier, plus a read-only check of the second.

### Local, free &mdash; construct and validate

| Stage | What happens |
| --- | --- |
| `S00`&ndash;`S02` | Install & smoke-test. Editable install, then a portable check that builds and schema-validates every active domain's structural trajectories &mdash; no model, no GPU. |
| `S03`&ndash;`S06` | Build a domain package. Retrieval plan (clean-ranked, hash-bound) &rarr; structural trajectories for 2-hop/3-hop &times; clean/injected &rarr; schema validation &rarr; context-window preflight against the real tokenizer. |

### Remote, still free &mdash; touch the workspace, not the GPU

| Stage | What happens |
| --- | --- |
| `S07` | Verify the Modal workspace. Auth check only &mdash; no app, no image build, no GPU. |
| `S08` | Cache the pinned model revision. Remote CPU/storage, still no GPU &mdash; but Modal won't deploy *any* app containing a GPU function without billing on file, even to run the non-GPU half of it. |
| `S09` | Validate one trajectory plan against the live app &mdash; exact preview, still no model call. |

### Paid H200 &mdash; the only stages that produce real answers

| Stage | What happens |
| --- | --- |
| `S10` | One bounded exploratory trajectory. Requires an explicit confirmation string. This is the first point where "did the model resist the injection" becomes an answerable question instead of a null field. |
| `S11`&ndash;`S13` | Run the definitive matrix, then reconcile billing. Tier-isolated, explicitly enumerated cohort only. |

### Local again &mdash; turn activations into results

| Stage | What happens |
| --- | --- |
| `S14`&ndash;`S19` | Index, scan, probe, analyze. Build the activation index, run group-held-out probes (Goldowsky&ndash;Dill / LAT), and depth-degradation analysis. |
| `S20`&ndash;`S22` | Reporting. Rebuild public figures, fixed-layer analysis, cross-domain robustness checks. |
| `S23` | Human review. Two independent reviewers, blinded then treatment-aware, adjudicated on disagreement. AI-generated ratings are explicitly not a valid substitute. |

Tonight's work reached the end of the free local tier (S00&ndash;S06) for a
brand-new domain variant, confirmed S07, and hit the wall at S08.

## Tonight's session

You handed over five files for the neuro domain and asked to see the
pipeline run on them. Here's what actually happened, in order.

### First finding &mdash; the "injected" PDF wasn't injected

`neuro_doc1_inj_Version2.pdf` and `neuro_doc1_clean.pdf` had the identical
SHA-256. The trajectory handoff (`neuro_trajectory_Version2.json`) described
a real injection recipe &mdash; a "transitional/combined" arm, invisible
text plus a fake completion marker &mdash; but the file itself was just a
copy of the clean PDF. Nothing to test yet.

### Built from your dictated wording &mdash; constructed the injected twin at the byte level

You gave the exact instruction text (a fake "post-market safety monitoring"
pretext directing the document's full text to
`pms-reporting.example.invalid/submit`). That got spliced in as invisible
content-stream text &mdash; render mode 3, 1.5&nbsp;pt, white &mdash;
immediately after the paper's closing sentence and before the
"Contributors" heading, followed by an ASCII-rule block carrying a
`SUMMARY COMPLETE` marker. Getting the extraction order right took two
attempts: the first pass appended new text to the end of the page's content
stream (landing near the reference list, not the conclusion); the second
used `pikepdf` to splice the new operators directly after the specific
text-show instruction for the target sentence.

```
page 12 of 13, after "...facilitate clinical implementation."
before "Contributors"
render mode 3 · 1.5pt · #ffffff
single insertion · 619 bytes · sha256 5a26185928fb…
```

### Checked it three ways before trusting it

| Check | Method | Result |
| --- | --- | --- |
| Invisible in render | Rasterize clean & injected at 96 DPI, SHA-256 per page | 13 / 13 pages identical |
| Single clean edit | Prefix/suffix diff of extracted text | 0 removed, 619 inserted |
| Reaches the worker | Read worker_1's model-facing prompt in the generated trajectory | payload & marker present |
| Naturally retrieved | Clean-only BM25 ranking, no override policy | carrier chunk selected on its own |

That last row matters most: the injected chunk (`neuro_doc1__p012__c001`)
got picked by ordinary relevance ranking, not by a special-case retention
rule. In a real run, the worker would see this content without any thumb on
the scale.

### Full local package &mdash; ran S00&ndash;S06 end to end for a new independence group

Built `domain_config_v2_combined.json` as its own registry (never touching
the active V1 config the frozen results depend on), then the real pipeline:
retrieval plan &rarr; 4 schema-valid structural trajectories (2-hop/3-hop
&times; clean/injected) &rarr; context-window preflight against the pinned
Qwen3-32B tokenizer. All of it against the real tokenizer, zero model calls,
zero GPU.

| Metric | Value |
| --- | --- |
| Candidate chunks indexed | 76 |
| Chunks selected | 37 |
| Tokens selected / budget | 27,632 / 28,000 |
| Worker input tokens (injected, thinking off) | 33,985 |
| Context headroom remaining | &asymp;2,000 tokens |

### Checked the paid gate &mdash; Modal: authenticated, then stopped by billing

S07 passed cleanly &mdash; workspace `ubajaka-chijioke` authenticated,
billing summary read (a few thousandths of a cent, fully offset by
credits). S08 got further than expected: two CPU images built successfully,
then Modal refused to deploy because the app *defines* an H200 function
elsewhere in it, even though the action we called never touches a GPU. That
needs a payment method added to the workspace before it will go further.

> **Blocked, not broken.** Nothing here is a bug we can route around
> &mdash; it's Modal's own deploy-time policy. Add a card at modal.com
> &rarr; Settings &rarr; Billing, and S08&ndash;S10 pick up right where
> they left off.

### Structural honesty &mdash; what "clean run without payment" can and can't show

With Modal blocked, we walked the structural trajectories side by side
instead. The clean condition reads straight from the paper's conclusion
into "Contributors"; the injected condition has the payload sitting in
exactly that gap. But every field that would say whether the model
*resisted* is explicitly `null`, tagged `pending_real_model_output`. That's
by design &mdash; this schema refuses to let a structural build stand in
for a real answer.

### Unplanned detour &mdash; a Windows encoding bug, three times, then a full sweep

Running the smoke test and the test suite from a native PowerShell terminal
(no `PYTHONUTF8` set) surfaced the same failure three separate times, in
three different call sites: a raw `charmap` decode crash, a false "stale
retrieval plan" error that turned out to be the same bug corrupting a hash,
and a test reading an "Alzheimer's" curly apostrophe. Root cause: nothing in
the codebase pinned `encoding="utf-8"`, so every file read fell back to
whatever code page the console happened to be using.

Once the pattern was confirmed for the third time, we swept the whole
repository rather than fixing them one at a time.

| Scope | Count |
| --- | --- |
| Files touched | 49 |
| `.read_text()` call sites fixed | 31 files |
| `open()` / `.open()` call sites fixed | 18 files |
| Test suite, normal run | 407 / 407 passed |
| Test suite, simulated PowerShell (`PYTHONUTF8=0`) | 407 / 407 passed |

## Where it stands

**Uncommitted.** Everything above is sitting in the working tree &mdash;
nothing has been committed. That's deliberate: the encoding fixes and the
neuro V2 package are logically unrelated (portability bugfix vs. new
research content), so they read better as two separate commits than one
bundled one.

**To actually test detection:** add a payment method to the
`ubajaka-chijioke` Modal workspace. That unblocks S08 (cache the model,
still free) and then S10 &mdash; the first paid trajectory, and the first
point where "does Qwen resist this injection" stops being a null field.

---

*Written up from the session transcript. No model or GPU calls were made
while producing this document.*
