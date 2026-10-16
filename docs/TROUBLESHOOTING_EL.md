# Αντιμετώπιση προβλημάτων

- **Λάθος Python:** απαιτείται 3.12.x· ελέγξτε `.venv\Scripts\python.exe --version`.
- **Dependency:** αναδημιουργήστε μόνο το νέο `.venv` από `requirements-dev.lock`.
- **Raw BLOCKED:** ορίστε `PV_THESIS_DATA_DIR` και ελέγξτε ονόματα/hashes ZIP.
- **Διακοπή training:** εκτελέστε `python scripts/resume_phase_2b.py`. Τα checkpoints/logs διατηρούνται.
- **RAM/δίσκος:** σταματήστε με ασφάλεια και κρατήστε το run directory.
- **LSTM difference:** ελέγξτε seed, CPU runtime, PyTorch και scientific diff· μη χρησιμοποιήσετε frozen ως νέα πρόβλεψη.
