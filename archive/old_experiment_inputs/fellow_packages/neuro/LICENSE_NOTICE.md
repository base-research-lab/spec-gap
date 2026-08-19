<!--
Renamed from: experiments/scenario1/inputs/fellow_packages/neuro/README.md
Original date: not encoded in the former filename
Renamed on: 2026-08-10
Purpose: preserve the third-party publication license boundaries for Neuro source documents.
-->

# Neuro source-document licenses

The repository's MIT License covers repository-authored code and metadata. It
does not relicense the third-party publication text in `documents/`; each
source remains governed by its publication license and attribution terms.

| Runtime ID | Source | License |
| --- | --- | --- |
| `neuro_doc1` | Feizpour et al., <https://doi.org/10.1016/j.ebiom.2024.105405> | CC BY 4.0 |
| `neuro_doc2` | Ashton et al., <https://doi.org/10.1002/alz.14621> | Creative Commons Attribution |
| `neuro_doc3` | Pahlke et al., <https://doi.org/10.1002/alz.70828> | Creative Commons Attribution-NonCommercial-NoDerivs |

In particular, reuse of `neuro_doc3_clean.txt` must remain non-commercial and
must not distribute modified or adapted content. The tracked text is an
attributed `pdftotext -raw` format shift of the source publication. The
controlled injection is applied only to `neuro_doc1`, which is CC BY 4.0;
`neuro_doc3` is never modified in either experimental arm.

The protocol-specific registry records the source URLs, DOIs, license labels,
checksums, and this redistribution decision. This notice is informational and
does not replace the source license terms.
