# Αναπαραγωγή αποτελεσμάτων φωτοβολταϊκών συστημάτων

Το αποθετήριο περιέχει τον κώδικα και τα απαραίτητα αρχεία για την επαλήθευση των αποτελεσμάτων μιας μελέτης φωτοβολταϊκών συστημάτων. Καλύπτει την πρόβλεψη παραγωγής, τη φυσική μοντελοποίηση και την παραμετρική ανάλυση διαφορετικών τεχνολογιών, ισχύων και κλιματικών συνθηκών.

Η βασική διαδικασία χρησιμοποιεί τα αποθηκευμένα δεδομένα αναφοράς στο `artifacts/approved_reference/`. Επανυπολογίζει τις μετρικές και δημιουργεί νέους πίνακες και αναφορές, χωρίς να απαιτεί επανεκπαίδευση των μοντέλων ή επανεκτέλεση όλων των πειραμάτων.

## Εγκατάσταση

Απαιτούνται Windows 11 και Python 3.12 (64-bit). Από ένα PowerShell terminal, στον φάκελο του repository:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --requirement requirements-dev.lock
.\.venv\Scripts\python.exe -m pip install --editable . --no-deps
```

## Χρήση

Έλεγχος του περιβάλλοντος και των διαθέσιμων αρχείων:

```powershell
.\.venv\Scripts\pv-thesis.exe doctor
```

Επαλήθευση όλων των αποτελεσμάτων:

```powershell
.\.venv\Scripts\pv-thesis.exe verify --all
```

Η επαλήθευση μπορεί να περιοριστεί σε ένα μόνο σκέλος:

```powershell
.\.venv\Scripts\pv-thesis.exe verify --branch forecasting
.\.venv\Scripts\pv-thesis.exe verify --branch physical
.\.venv\Scripts\pv-thesis.exe verify --branch parametric
```

Αναπαραγωγή των αποτελεσμάτων από τα δεδομένα αναφοράς και δημιουργία αναφοράς:

```powershell
.\.venv\Scripts\pv-thesis.exe reproduce --all --mode frozen
.\.venv\Scripts\pv-thesis.exe report
```

Οι νέες έξοδοι αποθηκεύονται στο `outputs/generated/consolidated-frozen/`.

Για την εκτέλεση των ελέγχων του κώδικα:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## Δεδομένα

Τα δεδομένα αναφοράς που απαιτούνται για την παραπάνω διαδικασία περιλαμβάνονται στο repository. Τα αρχικά δεδομένα που δεν μπορούν να διανεμηθούν δημόσια δεν περιλαμβάνονται. Περισσότερες πληροφορίες υπάρχουν στο `docs/DATA_ACCESS_EL.md`.

Μετά την εγκατάσταση των εξαρτήσεων, η επαλήθευση από τα αποθηκευμένα δεδομένα μπορεί να εκτελεστεί χωρίς σύνδεση στο διαδίκτυο και χωρίς GPU.
