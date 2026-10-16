# Αντιστοίχιση επιστημονικών αποτελεσμάτων

| Κλάδος | Αυθεντικός κώδικας | Frozen evidence | Υπολογισμός | Headline αποτέλεσμα |
|---|---|---|---|---|
| Forecasting | `src/forecasting_*`, `src/models`, `src/analysis/final_forecasting` | aligned predictions, splits, model predictions | site-macro RMSE/MAE/nRMSE, paired bootstrap | RF 11,8718%, SARIMA 12,3508%, LSTM 13,2290%, Persistence 14,3722% |
| Physical | `src/physical_benchmark` | 323.574 five-minute rows, predictions, calibrated parameters | component metrics και array-DC-normalized AC RMSE | 5,4093%–8,1986% |
| Parametric | `src/parametric_pv` | 144 scenarios, 1.261.440 hourly rows, TMY provenance | annual yield, clipping, invariance, variance decomposition | 1001,7–2079,5 kWh/kWp, climate 98,61% |

Οι configs βρίσκονται στα `config/forecasting`, `config/physical`, `config/parametric`. Τα import hashes βρίσκονται στο `artifacts/approved_reference/PROVENANCE.json`.
