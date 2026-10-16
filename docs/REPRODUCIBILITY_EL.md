# Αναπαραγωγιμότητα

Υποστηρίζονται τρία επίπεδα: (1) artifact/hash verification, (2) frozen-output metric recomputation, (3) προαιρετική πλήρης επιστημονική επανεκτέλεση. Η παρούσα παράδοση έχει ελεγχθεί για τα δύο πρώτα. **FULL EXPERIMENTAL RERUN: NOT TESTED** και για τους τρεις κλάδους.

Η εντολή `pv-thesis reproduce --all --mode frozen` διαβάζει frozen χρονοσειρές και υπολογίζει ξανά μετρικές χωρίς training, calibration ή simulation. Οι έξοδοι γράφονται χωριστά στο `outputs/generated/`.
