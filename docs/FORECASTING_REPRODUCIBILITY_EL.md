# Αναπαραγωγιμότητα πρόβλεψης

## Παγωμένο επιστημονικό πρωτόκολλο

- 11 συστήματα και 9 ανεξάρτητα sites.
- 934.800 canonical ωριαίες εγγραφές.
- 929.952 εγγραφές forecasting windows.
- Issue time στην τοπική μεσονύκτια αρχή της ημέρας στόχου.
- D+1 ορίζοντας 24 ωρών.
- Αποκλεισμός ημερών DST με 23/25 τοπικές ώρες.
- Τέσσερα expanding-window folds, validation 90 ημερών και ελάχιστο train 730 ημερών.
- Τελικό test 365 ημερών.
- Κοινή daylight μάσκα 36.529 παρατηρήσεων.
- Πρώτα υπολογισμός ανά φυσικό site και έπειτα site-macro aggregation.

## Modes

Το frozen mode επανυπολογίζει τις μετρικές και το paired moving-block bootstrap χωρίς retraining. Το canonical mode δημιουργεί νέα windows και εκπαιδεύει όλα τα μοντέλα από το approved canonical dataset. Το raw mode εκτελεί επιπλέον audit και canonicalization από τα οκτώ ιδιωτικά ZIP.

Καμία κατάσταση `BLOCKED` ή `NOT TESTED` δεν μετατρέπεται σε `PASS`. Τα νέα outputs γράφονται αποκλειστικά σε νέο run directory.

## Scientific comparison

Οι νέες προβλέψεις ενώνονται με τις frozen με τα κλειδιά `system_id`, `target_timestamp_utc`, `horizon`. Απορρίπτονται διπλότυπα, ελλείποντα και επιπλέον κλειδιά. Αναφέρονται observation counts, masks, per-site και site-macro μετρικές, αριθμητικές αποκλίσεις, κατάταξη και στατιστικές συγκρίσεις.

Η ισότητα bitwise δεν θεωρείται προϋπόθεση για LSTM retraining. Ωστόσο καμία ανοχή δεν επιλέγεται εκ των υστέρων ώστε να μετατρέψει αποτυχία σε επιτυχία.
