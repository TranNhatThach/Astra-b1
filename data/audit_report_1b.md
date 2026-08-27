# ASTRA-1B — PHASE 7C FULL-SCALE CORPUS AUDIT REPORT

- **Dataset Version:** `astra-test-v1`
- **Status:** **READY_FOR_ASTRA_1B**
- **Total Tokens:** `16,384` tokens
- **Total Sequences:** `4` ($T=4096$)
- **Unique Documents:** `76`
- **Dataset Hash:** `525339bc2e16933f942cbd890408f75f24e83c8c6556ee2c01962ff985738423`
- **Tokenizer Hash:** `514a02f5e8a4eb88b3113c22e022fb1969acddbbf9487f261f615b6e384dc5e8` (`FROZEN`)

## 1. Token-Level Mixture Accounting

| Category | Target % | Actual % | Actual Tokens | Deviation (pp) |
| :--- | :--- | :--- | :--- | :--- |
| **Web** | 45.0% | 51.68% | 8,571 | +6.68pp |
| **Educational** | 15.0% | 13.87% | 2,301 | -1.13pp |
| **Code** | 15.0% | 11.5% | 1,907 | -3.50pp |
| **Math** | 10.0% | 8.7% | 1,442 | -1.30pp |
| **Vietnamese** | 10.0% | 7.25% | 1,203 | -2.75pp |
| **Dialogue** | 5.0% | 6.99% | 1,160 | +1.99pp |

## 2. Document Diversity & Quality Metrics

- Raw Documents Seen: 80
- Unique Documents Accepted: 76
- Exact Duplicate Rate: 0.00%
- Near Duplicate Rate: 5.00%
- Quality Rejection Rate: 0.00%
- Mean Tokens / Document: 218.2
- Median Tokens / Document: 212.5

## 3. Generated Shards

| Shard Name | Sequences | Tokens | SHA-256 Checksum |
| :--- | :--- | :--- | :--- |
| `shard-000000.bin` | 4 | 16,384 | `d100b7f5a1c0c0711aaa525a35214f471fc2dd3dbd5227cc0caff63ba5f96136` |
