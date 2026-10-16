# Raw αναπαραγωγή

Απαιτεί τα εξουσιοδοτημένα τοπικά ZIP στο `PV_THESIS_DATA_DIR`. Εκτελεί audit, canonicalization, preprocessing, splits, νέα εκπαίδευση και αξιολόγηση. Τα raw δεδομένα δεν αντιγράφονται ούτε ανεβαίνουν.

```powershell
$env:PV_THESIS_DATA_DIR = 'D:\private\pv-forecasting-data'
.\.venv\Scripts\pv-thesis.exe forecasting reproduce --mode raw --confirm-full-training --output outputs\generated\raw-resumable
```

Αποτυχία πρόσβασης/hash δηλώνεται BLOCKED/FAIL· δεν γίνεται σιωπηρή μετάβαση στο canonical dataset.
