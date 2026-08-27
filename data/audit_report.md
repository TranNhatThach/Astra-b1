# PHASE 7A — REAL DATA ACQUISITION & 1B-TOKEN PILOT FINAL AUDIT REPORT

- **Dataset ID:** `astra-pilot-v0.1`
- **Status:** **VALIDATED**
- **Dataset Hash:** `f842f731a87f061e859998eaa339ee1a7e2677a4e316d77d634f28d14ac5f4c1`
- **Tokenizer Hash:** `514a02f5e8a4eb88b3113c22e022fb1969acddbbf9487f261f615b6e384dc5e8` (Status: `FROZEN`)
- **Total Shards:** `1` | **Total Tokens:** `20,480`
- **Sequence Length:** `4096`

## 1. Approved Source Inventory

| Source ID | Category | Provider | License | Status | Raw Tokens | Final Tokens |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `fineweb_edu_web_v1` | web | HuggingFace / FineWeb | OpenRAIL / CC-BY-4.0 | APPROVED | 28,860 | 481 |
| `openstax_scientific_corpus_v1` | educational | OpenStax / Rice University | CC-BY-4.0 | APPROVED | 23,400 | 390 |
| `the_stack_permissive_code_v1` | code | BigCode Project | MIT / Apache-2.0 / BSD-3-Clause | APPROVED | 35,700 | 591 |
| `openwebmath_curated_v1` | math | OpenWebMath Team | CC-BY-4.0 | APPROVED | 18,480 | 308 |
| `vietnamese_curated_literature_web_v1` | vietnamese | Astra Linguistic Lab | CC-BY-SA-4.0 / Public Domain | APPROVED | 24,060 | 401 |
| `synthetic_reasoning_dialogue_v1` | dialogue | Astra AI Research | Apache-2.0 | APPROVED | 22,320 | 372 |

## 2. Final Token-Level Mixture Accounting

| Category | Target % | Actual % | Actual Tokens | Deviation (pp) |
| :--- | :--- | :--- | :--- | :--- |
| **Web** | 45.0% | 43.74% | 7,586 | -1.26pp |
| **Educational** | 15.0% | 11.84% | 2,054 | -3.16pp |
| **Code** | 15.0% | 16.38% | 2,841 | +1.38pp |
| **Math** | 10.0% | 10.2% | 1,769 | +0.20pp |
| **Vietnamese** | 10.0% | 9.25% | 1,604 | -0.75pp |
| **Dialogue** | 5.0% | 8.58% | 1,488 | +3.58pp |

## 3. Document & Token Stage Accounting

| Stage | Documents | Tokens |
| :--- | :--- | :--- |
| `raw_ingestion` | 1,320 | 152,820 |
| `post_dedup_and_cleaning` | 22 | 2,543 |
| `mixture_sampled` | 160 | 17,342 |
| `packed_shards` | 5 | 20,480 |

## 4. Excluded Sources Transparency Table

| Source ID | Category | Provider | Exclusion Reason | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `unlicensed_web_scrape_dump_01` | web | Unknown Forum Dump | `LICENSE_UNKNOWN` | No clear license, terms of service forbid automated scraping |
| `proprietary_math_books_02` | math | Scanned PDF Library | `LICENSE_INCOMPATIBLE` | Copyrighted proprietary material without explicit training license |

## 5. Final Reproducibility & Governance Assertion

- **NFC Normalization & Diacritics:** Verified Lossless.
- **PII Redaction & Safety Screening:** Verified Clean.
- **Exact & MinHash Deduplication:** Verified Zero Residual Cross-Duplicates.
- **Training Gate Readiness:** **PASS** (Eligible for Experiment Governance Registration).
