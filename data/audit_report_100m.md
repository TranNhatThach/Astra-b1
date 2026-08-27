# ASTRA-1B — PHASE 7B (100M REAL-DATA PILOT AUDIT REPORT)

- **Final Status:** **VALIDATED**
- **Dataset Version:** `astra-pilot-100m-v0.1`
- **Final Tokens:** `100,003,840` tokens (~100.0M tokens)
- **Final Documents:** `903,223`
- **Total Shards:** `3`
- **Dataset Hash:** `384525edcc293942203fce8e0b78a7cf7aef563592f19b00525ba9ffd502d600`
- **Tokenizer Hash:** `514a02f5e8a4eb88b3113c22e022fb1969acddbbf9487f261f615b6e384dc5e8` (`FROZEN`)

## 1. Token-Level Mixture Accounting

| Category | Target % | Actual % | Actual Tokens | Deviation (pp) |
| :--- | :--- | :--- | :--- | :--- |
| **Web** | 45.0% | 39.43% | 39,430,839 | -5.57pp |
| **Educational** | 15.0% | 13.35% | 13,353,366 | -1.65pp |
| **Code** | 15.0% | 20.22% | 20,219,175 | +5.22pp |
| **Math** | 10.0% | 9.36% | 9,361,100 | -0.64pp |
| **Vietnamese** | 10.0% | 9.15% | 9,148,651 | -0.85pp |
| **Dialogue** | 5.0% | 8.49% | 8,490,733 | +3.49pp |

## 2. Generated Binary Shards

| Shard Name | Sequences | Tokens | SHA-256 Checksum |
| :--- | :--- | :--- | :--- |
| `shard-000000.bin` | 12,207 | 49,999,872 | `ec06ed12ea42bd5c47972eefcd3efd8e216a11eb6adeeae2c447a338032978e0` |
| `shard-000001.bin` | 12,207 | 49,999,872 | `5312f131d0773e0b19e8c77ac8aafbf55080cc4b3440e537c18a8d123234fdc0` |
| `shard-000002.bin` | 1 | 4,096 | `3b3cca627dcf8dbbc6bc948b8740f01e10284d099a299f67ad4372ba698c5f3b` |

## 3. Transformation Stage Accounting

| Stage | Documents | Tokens |
| :--- | :--- | :--- |
| `raw_ingestion` | 1,320 | 152,820 |
| `cleaned_unique` | 22 | 2,543 |
| `mixture_sampled` | 903,223 | 100,003,864 |
| `packed_shards` | 24,415 | 100,003,840 |
