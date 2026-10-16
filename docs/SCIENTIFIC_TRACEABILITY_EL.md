# Επιστημονική ιχνηλασιμότητα

## Προέλευση

Τα directories `src/data_audit`, `src/forecasting_data`, `src/forecasting_experiment`, `src/models/{sarima,random_forest,lstm_baseline}` και `src/analysis/final_forecasting` είναι αντίγραφα της ανακτημένης αυθεντικής υλοποίησης. Τα hashes τους επαληθεύτηκαν πρώτα έναντι του Phase 2A manifest και ξανά μετά την εισαγωγή.

Το `artifacts/approved_reference/PROVENANCE.json` καταγράφει source path, imported path, SHA-256, μέγεθος και ερευνητικό ρόλο για κάθε εισαγόμενο αρχείο. Οι clean-room wrappers βρίσκονται μόνο στο `src/pv_thesis`.

## Αλυσίδα

`raw ZIP` → dataset audit → canonical hourly dataset → local-day windows και frozen splits → Persistence/SARIMA/RF/LSTM v1.1 → stable-key alignment → common daylight mask → per-site metrics → site-macro metrics → paired block bootstrap.

## Κρίσιμα συμβόλαια

- Δεν χρησιμοποιούνται observations της target day πριν από το issue time.
- Δεν γίνεται random time shuffle.
- Το test set δεν συμμετέχει σε model selection.
- Δεν αλλάζουν imputation, scaling, daylight mask, DST gate ή selected candidates.
- LSTM: 168 ώρες ιστορίας προς 24 ώρες output, seed 42 και τελικό epoch count από το παγωμένο CV.
- RF: deterministic/historical features μόνο, seed 42.
- SARIMA: πέντε preregistered candidates και state update `append(refit=False)`.
- Bootstrap: paired ανά site και ημέρα, moving blocks 7 ημερών.
