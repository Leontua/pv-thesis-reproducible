# ETSIDI Six-Technology Physical Model Benchmark v1.0

**Status: FROZEN.** Scope is exclusively T1-S1 through T1-S6. No imputation was used.

The deterministic chain is measured 5-minute in-plane irradiance/weather → calibrated Faiman module temperature → locked-datasheet De Soto single-diode DC → one calibration-only DC retention scalar → 1200 W PVWatts inverter.

## Frozen dataset and split

- Long-format rows: 323,574
- Timestamp range: 2025-05-02 06:00:00 to 2026-03-31 21:55:00
- Unique local days: 290
- Calibration days: 176
- Test days: 57
- Validation days: 57

## Calibration-only parameters

| System | Faiman U0 | Faiman U1 | DC retention | DC loss | Thermal source |
|---|---:|---:|---:|---:|---|
| T1-S1 | 33.821 | 1.812 | 0.9128 | 8.72% | system calibration |
| T1-S2 | 36.165 | 1.897 | 0.9186 | 8.14% | system calibration |
| T1-S3 | 34.068 | 2.179 | 0.8336 | 16.64% | system calibration |
| T1-S4 | 34.068 | 2.179 | 0.5343 | 46.57% | median transfer; no Tmodule sensor |
| T1-S5 | 28.383 | 2.806 | 0.7947 | 20.53% | system calibration |
| T1-S6 | 39.995 | 3.013 | 0.8990 | 10.10% | system calibration |

## Frozen test results

| System | DC nMAE | DC nRMSE | AC nMAE | AC nRMSE |
|---|---:|---:|---:|---:|
| T1-S1 | 3.779% | 8.158% | 3.590% | 7.648% |
| T1-S2 | 3.548% | 8.486% | 3.193% | 7.121% |
| T1-S3 | 6.234% | 9.299% | 5.180% | 7.857% |
| T1-S4 | 3.231% | 5.401% | 2.368% | 3.494% |
| T1-S5 | 4.105% | 7.796% | 3.551% | 7.103% |
| T1-S6 | 3.422% | 8.090% | 3.676% | 8.140% |

## Interpretation guardrails

- T1-S4 has no module-temperature sensor; its Faiman U0/U1 are the medians of the five instrumented systems and no temperature score is reported.
- T1-S4 and T1-S5 store one-string DC current; the locked ×2 topology multiplier reconstructs total array DC current before any calibration.
- The single DC scalar is fitted only on calibration days and is never re-fitted on validation or test.
- PVWatts inverter parameters are fixed from the 1200 W Sunny Boy nameplate/efficiency; measured DC is also passed through PVWatts for isolated inverter-component validation.
- The test partition is used exactly once for the frozen report.
