# Ενοποιημένο λογισμικό διπλωματικής φωτοβολταϊκών συστημάτων

## 1. Περιγραφή της διπλωματικής εργασίας

Το repository συγκεντρώνει το επιστημονικό λογισμικό της διπλωματικής στο ΕΜΠ, Σχολή ΗΜΜΥ, για επιβλέπουσα τη Νίκη Γκόλια. Μελετά πρόβλεψη παραγωγής, φυσική επικύρωση Φ/Β μοντέλων και παραμετρική επίδραση τεχνολογίας, ισχύος και κλίματος.

## 2. Σκοπός του λογισμικού

Σκοπός είναι η εγκατάσταση, επιθεώρηση και ανεξάρτητη επαλήθευση των υπαρχόντων αποτελεσμάτων. Το βασικό workflow επανυπολογίζει μετρικές από frozen artifacts· δεν εκπαιδεύει, δεν βαθμονομεί και δεν εκτελεί εκ νέου τα 144 σενάρια.

## 3. Επιστημονικά σκέλη

- **Πρόβλεψη παραγωγής:** 11 systems, 9 sites, Persistence, SARIMA, Random Forest, LSTM v1.1.
- **Φυσική μοντελοποίηση:** έξι ETSIDI arrays, Faiman → CEC/SAM single diode → DC retention → PVWatts inverter, Mode A/Mode B.
- **Παραμετρική ανάλυση:** 6 τεχνολογίες × 4 ισχείς × 6 κλίματα = 144 σενάρια.

## 4. Περιεχόμενα αποθετηρίου

Τα `src/`, `config/`, `tests/`, `artifacts/approved_reference/`, `docs/`, `reports/` και `outputs/generated/` χωρίζουν κώδικα, ρυθμίσεις, immutable frozen evidence, τεκμηρίωση και νέες εξόδους. Το `PROVENANCE.json` συνδέει κάθε εισαγόμενο αρχείο με την αυθεντική πηγή ή package.

## 5. Απαιτήσεις υπολογιστή

Ελεγμένο περιβάλλον: Windows 11 x86_64, Python 3.12.14, CPU. Η frozen επαλήθευση δεν απαιτεί GPU ή πολυώρη εκτέλεση.

## 6. Εγκατάσταση Python και βιβλιοθηκών

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --requirement requirements-dev.lock
.\.venv\Scripts\python.exe -m pip install --editable . --no-deps
```

## 7. Προετοιμασία δεδομένων

Τα εγκεκριμένα frozen artifacts βρίσκονται στο `artifacts/approved_reference/`. Raw δεδομένα με αβέβαια δικαιώματα δεν περιλαμβάνονται για δημόσια διανομή. Οδηγίες και hashes: `docs/DATA_ACCESS_EL.md`.

## 8. Γρήγορη έναρξη για την επιβλέπουσα

Οι ακόλουθες εντολές έχουν εκτελεστεί επιτυχώς:

```powershell
.\.venv\Scripts\pv-thesis.exe doctor
.\.venv\Scripts\pv-thesis.exe verify --all
.\.venv\Scripts\pv-thesis.exe reproduce --all --mode frozen
.\.venv\Scripts\pv-thesis.exe report
.\.venv\Scripts\python.exe -m pytest
```

## 9. Επαλήθευση επιστημονικών αποτελεσμάτων

Το `verify --all` ελέγχει hashes, πληθυσμούς, splits, normalization, παραμέτρους και ανεξάρτητους αριθμητικούς υπολογισμούς. Επιτρέπεται και `--branch forecasting|physical|parametric`.

## 10. Αναπαραγωγή αποτελεσμάτων forecasting

Το frozen workflow υπολογίζει ξανά site-macro RMSE/MAE/nRMSE και paired bootstrap από 36.529 κοινές παρατηρήσεις. Αποτελέσματα: RF 11,87180843%, SARIMA 12,35076222%, LSTM 13,22902090%, Persistence 14,37215958%.

## 11. Αναπαραγωγή φυσικής επικύρωσης

Οι frozen measured/modelled χρονοσειρές επαναξιολογούνται χωρίς refit. Το end-to-end AC test nRMSE με array DC nameplate normalization είναι 5,409319%–8,198587% για έξι arrays.

## 12. Αναπαραγωγή παραμετρικών αποτελεσμάτων

Από 1.261.440 υπάρχουσες hourly εγγραφές υπολογίζονται ξανά annual yield και losses. Yield: 1001,7068–2079,5146 kWh/kWp/year· climate share 98,6115%· technology share 1,2723%.

## 13. Παραγωγή πινάκων και σχημάτων

Το frozen workflow γράφει επανυπολογισμένους πίνακες και verification manifests στο `outputs/generated/consolidated-frozen/`. Τα εγκεκριμένα αρχικά σχήματα παραμένουν στο reference tree.

## 14. Πλήρης επανεκτέλεση πειραμάτων ως προαιρετική δυνατότητα

Ο αυθεντικός κώδικας training/calibration/simulation διατηρείται για μελλοντική προαιρετική χρήση. **FULL EXPERIMENTAL RERUN: NOT TESTED.** Δεν αποτελεί μέρος του γρήγορου workflow.

## 15. Γνωστοί περιορισμοί

Δεν διανέμονται raw δεδομένα χωρίς έλεγχο αδειών. Frozen reproduction δεν ισοδυναμεί με νέο training. Προγενέστερη μερική canonical SARIMA απόπειρα σταμάτησε μετά την αλλαγή πολιτικής και τα checkpoints διατηρούνται μόνο ως ιστορικό εκτέλεσης.

## 16. Ακαδημαϊκή τεκμηρίωση

Το `docs/SCIENTIFIC_RESULTS_MAPPING_EL.md` αντιστοιχίζει source → configuration → frozen evidence → υπολογισμό → αποτέλεσμα. Οι πειραματικές διακρίσεις site/system, Mode A/Mode B και capacity normalization παραμένουν ρητές.

## 17. Προέλευση δεδομένων και άδειες

Τα manifests περιέχουν αναμενόμενα paths, SHA-256, providers όπου είναι διαθέσιμοι και redistribution status. Πριν από GitHub απαιτείται τελικός νομικός/αδειοδοτικός έλεγχος.

## 18. Οδηγίες αντιμετώπισης προβλημάτων

Δείτε `docs/TROUBLESHOOTING_EL.md`. Χρησιμοποιήστε Python 3.12.x, εγκαταστήστε από το lockfile και εκτελέστε πρώτα `pv-thesis doctor`. Μην εκτελείτε historical training scripts για να ελέγξετε την εγκατάσταση.
