# Περιβάλλον εκτέλεσης

## Ανακτημένο περιβάλλον

| Στοιχείο | Έκδοση |
|---|---|
| Python | 3.12.14, CPython, Windows x86_64 |
| NumPy | 2.5.3 |
| pandas | 3.0.6 |
| SciPy | 1.18.1 |
| scikit-learn | 1.5.2 |
| statsmodels | 0.15.0 |
| PyTorch | 2.5.1+cpu |
| pyarrow | 25.0.1 |
| matplotlib | 3.11.2 |
| PyYAML | 6.0.3 |
| pvlib | 0.13.1 |
| joblib | 1.4.2 |

Οι εκδόσεις προέρχονται από τα πραγματικά `.dist-info`, τα run manifests και το Python executable του αυθεντικού workspace. Το `torch==2.5.1` επιλύεται σε CPU build `2.5.1+cpu` στα Windows.

## Εγκατάσταση

```powershell
<Python-3.12.14> -m venv .venv
.\.venv\Scripts\python.exe -m pip install --requirement requirements-dev.lock
.\.venv\Scripts\python.exe -m pip install --editable . --no-deps
.\.venv\Scripts\pv-thesis.exe doctor
```

Η εγκατάσταση δεν επιτρέπεται να αναβαθμίσει σιωπηρά τις επιστημονικές εξαρτήσεις: όλες είναι pinned στο `requirements.lock`.

## Υπολογιστικοί πόροι και ντετερμινισμός

- LSTM v1.1: CPU, seed 42, 4 torch threads, seeded epoch permutations, χωρίς απαίτηση CUDA.
- Random Forest: `random_state=42`, `n_jobs=-1`, χωρίς temporal shuffle.
- Final bootstrap: seed 20261006, 10.000 replicates, paired moving blocks 7 ημερών.
- SARIMA: statsmodels 0.15.0, `lbfgs`, έως 200 iterations, `append(refit=False)` ανά ημέρα.

Μικρές αριθμητικές αποκλίσεις μπορεί να προκύψουν από BLAS/OpenMP, διαφορετικό αριθμό threads ή διαφορετικό λειτουργικό. Δεν θεωρούνται αυτόματα αποδεκτές: καταγράφονται και αξιολογούνται ανά μετρική, site και prediction key.

Η πλήρης canonical εκτέλεση εκτιμάται σε 36–120 ώρες CPU, τουλάχιστον 5 GB ελεύθερου δίσκου και 16 GB RAM· συνιστώνται 10 GB και 32 GB αντίστοιχα. Η εκτίμηση βασίζεται στα αυθεντικά logs, όχι σε εγγύηση benchmark.
