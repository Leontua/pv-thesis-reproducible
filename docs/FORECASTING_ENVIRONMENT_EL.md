# Περιβάλλον πρόβλεψης

## Αυθεντικό περιβάλλον

Τα Phase 2A τεκμήρια κατέγραψαν Python 3.12.14, NumPy 2.5.3, pandas 3.0.6, SciPy 1.18.1, scikit-learn 1.5.2, statsmodels 0.15.0, PyTorch 2.5.1+cpu, pyarrow 25.0.1, matplotlib 3.11.2, PyYAML 6.0.3, pvlib 0.13.1 και joblib 1.4.2 σε Windows 11.

## Νέο συμβατό περιβάλλον

Το `.venv` χρησιμοποιεί τις ίδιες επιστημονικές εκδόσεις. Το distribution metadata του PyTorch είναι `2.5.1`, ενώ `torch.__version__` αναφέρει `2.5.1+cpu` και `torch.cuda.is_available()` είναι false. Προστέθηκαν μόνο `tzdata==2026.5` και `pytest==8.3.5` για φορητή ζώνη ώρας και tests. Οι ακριβείς απαιτήσεις είναι στα `requirements.lock` και `requirements-dev.lock`.
