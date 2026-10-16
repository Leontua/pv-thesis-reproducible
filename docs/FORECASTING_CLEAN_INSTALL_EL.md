# Καθαρή εγκατάσταση

Απαιτείται Python 3.12.x 64-bit σε Windows. Από καθαρό αντίγραφο:

```powershell
$Python312 = 'C:\Path\To\Python312\python.exe'
& $Python312 --version
& $Python312 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --requirement requirements-dev.lock
.\.venv\Scripts\python.exe -m pip install --editable . --no-deps
.\.venv\Scripts\pv-thesis.exe doctor
.\.venv\Scripts\pv-thesis.exe forecasting verify
.\.venv\Scripts\pv-thesis.exe forecasting reproduce --mode frozen
.\.venv\Scripts\python.exe -m pytest
```

Το `--no-deps` διατηρεί τις εκδόσεις του lock file. Το αποτέλεσμα του clean-install test καταγράφεται στο `PROJECT_PROGRESS.json`.
Χρησιμοποιήστε ρητή διαδρομή προς τον interpreter όταν το `python` στο PATH είναι Windows Store alias ή διαφορετική έκδοση. Η αρχική εγκατάσταση απαιτεί package index ή τοπικό pip cache· οι frozen εντολές μετά την εγκατάσταση λειτουργούν χωρίς δίκτυο.
