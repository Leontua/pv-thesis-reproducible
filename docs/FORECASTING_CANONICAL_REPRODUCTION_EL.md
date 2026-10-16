# Canonical αναπαραγωγή

Ξεκινά από το εγκεκριμένο `core_hourly_long.parquet`, δημιουργεί windows/splits, εκπαιδεύει Persistence, SARIMA, Random Forest και LSTM v1.1 και εκτελεί κοινή αξιολόγηση.

```powershell
.\.venv\Scripts\pv-thesis.exe forecasting reproduce --mode canonical --confirm-full-training --output outputs\generated\canonical-resumable
```

Η εκτέλεση είναι resumable και διατηρεί logs/checkpoints. PASS δίνεται μόνο μετά από όλα τα στάδια. Frozen προβλέψεις δεν υποκαθίστανται ποτέ ως νέες.
