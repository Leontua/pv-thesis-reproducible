# Frozen αναπαραγωγή

Το `pv-thesis forecasting reproduce --mode frozen` διαβάζει τις εγκεκριμένες προβλέψεις, ελέγχει stable keys και επανυπολογίζει site, site-macro και paired-bootstrap αποτελέσματα. Δεν γίνεται training. Επιτυχία σημαίνει αριθμητική αναπαραγωγή παγιωμένων μετρικών, όχι ανεξάρτητη ανακατασκευή προβλέψεων. Το manifest είναι `outputs/generated/frozen-resumable/run_manifest.json`.
