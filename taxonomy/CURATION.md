# Taxonomy curation log

**Input:** 24 clusters from UMAP + HDBSCAN over 529 LLM-extracted issue statements (39 noise), each named by Claude
(`taxonomy_draft.json`). **Output:** `taxonomy.json` — 5 level-1 groups, 19 level-2 themes, used for classification.

## Why curate at all
- Clusters are only moderately stable across UMAP seeds (pairwise ARI mean 0.55, min 0.43; 19–25 clusters).
- The LLM's suggested parents were inconsistent (17 different level-1 names, e.g. "Identity" vs "Identity & Authentication").
- One real-world theme was split across many clusters (the Detect outage spans 5), while some real themes landed in noise.

## Decisions
| Change | Clusters | Reason |
|---|---|---|
| Merge into *Outage root cause & remediation* | 21, 22, 23 | Same technical story at different levels of detail |
| Merge into *Outage business impact & trust* | 13, 14 | Customer-side consequences of the same outage |
| Merge into *Incident communication* | 19, 20 | Duplicate themes |
| Merge into *Detect alert quality* | 17, 18 | Latency and false positives are both "can I trust the alerts" |
| Merge into *Identity authentication & access* | 15, 16 | SSO and MFA issues come up together, same buyer |
| Merge into *Compliance framework coverage* | 2, 3, 9 | HIPAA / PCI clusters are framework instances, too small alone |
| Merge into *Compliance report quality & performance* | 10, 12 | Both are "the report doesn't work as needed" |
| **Add** *Integrations & connectors* | — | Recurring across support and competitive calls but never its own cluster |
| **Add** *Competitive pressure* | split from 13 | Sales needs competitor signal independent of the outage |
| **Add** *Release quality & regressions* | from 15 + noise | QA gaps and remediation side effects are an engineering signal |

Design rules: level-1 groups map to an owner (Eng/SRE, Product, Sales/Finance, CS/Support, Eng leadership);
level-2 themes are specific enough to act on and broad enough to have ≥5 calls; calls get 1–4 labels.
