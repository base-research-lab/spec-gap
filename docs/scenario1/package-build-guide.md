# Scenario 1: building a domain package

This is the current authoring guide for Scenario 1 domain packages. The
original Word document is preserved at
[`scenario1_package_build_guide.md.docx`](scenario1_package_build_guide.md.docx).

## What a domain package contains

Build one independent match group per domain. Each group contains five files:

1. three clean documents;
2. one injected document; and
3. one package-level trajectory handoff file.

The three clean documents are the authentic working package that a person in
the field would plausibly retrieve for the task. The injected document is a
twin of one clean document: outside the hidden instruction, its content and
structure must be identical to the clean source.

The clean condition uses the three clean documents. The injected condition
uses the same package except that the selected clean carrier is replaced by
its injected twin. This creates a presence-matched document pair in which the
injection is the only intended change.

The four content files go in the match-pair folder. The package-level
trajectory handoff goes in the trajectory folder.

## Package handoff versus generated trajectory

The fifth file is a package handoff: it describes the domain, task, source
documents, injected twin, and four conditions before any model run occurs. It
is not a completed model-run trajectory and must not contain fabricated model
outputs, behavioral labels, action results, token IDs, or activation paths.

The pipeline later creates one formal `spec_gap.scenario1.v2` execution record
for each clean/injected and 2-hop/3-hop condition. The generated-record
contract is documented in the [trajectory schema guide](schema.md).

## Required handoff metadata

The package-level trajectory handoff must identify:

- `domain_id`, `task_family_id`, and `independence_group_id`;
- `task.user_task` and `task.expected_benign_behavior`;
- every document's ID, title, and filename;
- the one document whose `role` is `injection_carrier`;
- `injection.insertion_anchor`;
- `provenance.created_by`; and
- the original, clean, and injected PDF filenames when those source PDFs
  exist.

Do not invent PDF filenames for an inline-only source. Preserve the real
source filename or omit the optional PDF-provenance field.

Older handoff skeletons that use only `group_id`, `domain`, and bare filename
arrays are deprecated. They do not preserve the independence, document-role,
insertion, or provenance fields required by the current workflow.

## Naming

Give both Drive folders one subfolder per domain. Name every file so it remains
self-identifying outside its folder because the trajectory handoff references
documents by filename.

Use:

```text
<domain-short-code>_doc<slot>_<clean-or-inj>.<extension>
<domain-short-code>_trajectory.json
```

Use the same slot number for an injected twin and the clean document it
shadows. Use the same domain short code as `domain_id`.

For example:

```text
petro_doc1_clean.pdf
petro_doc2_clean.png
petro_doc3_clean.docx
petro_doc2_inj.png
petro_trajectory.json
```

Here, the injected file shadows `petro_doc2_clean.png`; the trajectory handoff
must record that relationship.

Current short codes include:

```text
petro  telecom  neuro  aihc  kg  convex  macro  fin  policy
```

## What the three clean documents should be

Choose document types that are authentic to the field, not generic filler.
The three documents should read as a genuine set that someone would retrieve
together for one task. A finance package, for example, might contain a
corporate ledger, a budget sheet, and an acquisition plan. A neuroscience
package might contain a biomarker assay write-up, a cohort table, and a
methods memo.

Two requirements are non-negotiable:

1. Mix plain text and mixed media. Do not submit three plain prose files. At
   least one document must contain realistic structure such as a table,
   figure, form, or embedded media.
2. Make every document full-length and realistic. Thin abstracts are not
   sufficient. A reader from the field should not be able to identify the
   package as benchmark filler.

## Building the injected twin

Start from an exact copy of one clean document. Add the instruction with a
concealment method appropriate to that document's medium. The user task remains
benign; the instruction belongs in the document body or rendered content,
never in the user task or file metadata.

Methods used so far include invisible white text in a report body, text carried
only in a rasterized figure, and a payload framed as an audit or compliance
note in the running text.

Placement matters as much as concealment. Worker_1 receives the whole
document, so the payload may appear anywhere `pdftotext -raw` actually
extracts as text — running text, a figure caption, or a references section —
but a raster-only payload with no underlying text layer will never reach the
worker. Do not lead with it so plainly that a person scanning the file would
immediately notice it. Record the deterministic insertion location as
`injection.insertion_anchor`.

## Quality checks

- Use a concealment method and wording that are unique to the group. Reusing
  an attack surface across domains can let a probe learn the framing instead
  of the compromise.
- Keep `injection_family` and `injection_placement` fixed across a multi-domain
  batch, but use a domain-specific `injection.carrier_marker`. Allowing each
  domain its own plausible compliance framing intentionally relaxes the older
  shared-marker invariant.
- Confirm `pdftotext -raw` can actually expose the payload. A raster-only
  injection is not a valid injected condition if the pipeline extracts text
  only.
- Verify that the injected twin differs from its clean source only by the
  payload. Keep the same title, surrounding content, and structure.
- Do not warn downstream agents about injected instructions in the planner
  prompt. That presupposes the threat and biases the run toward resistance.
- Make tasks and documents independent across groups. Near-duplicate language
  or attack surfaces can leak across train, validation, and test splits.

## Open-access sources

- Prefer material licensed under CC BY or CC0.
- Suitable sources include publishers that use CC BY by default, the PMC Open
  Access commercial-use subset, and relevant DOAJ-listed journals.
- Avoid Non-Commercial licenses when the intended use could be commercial.
- Record the original authors, journal, and source in project documentation.
- Respect host API terms, rate limits, and automated-download restrictions.

## Final package check

Before handoff, confirm that:

- there are exactly three clean content files, one injected twin, and one
  trajectory handoff;
- filenames identify the domain, slot, and clean/injected condition;
- the injected file clearly identifies which clean file it shadows;
- all required identity, task, document, injection, and provenance metadata is
  present; and
- the clean and injected packages differ only by the carrier swap.
