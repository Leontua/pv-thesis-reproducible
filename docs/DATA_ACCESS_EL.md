# Πρόσβαση δεδομένων

Τα frozen evidence που απαιτούνται για το γρήγορο verification περιλαμβάνονται μόνο όπου επιτρέπεται η τοπική παράδοση. Τα πρωτογενή forecasting ZIP, τα αρχικά ETSIDI inputs και οποιαδήποτε δεδομένα με αβέβαιη άδεια δεν προορίζονται για αυτόματη δημοσίευση.

- Forecasting raw: εξωτερική πρόσβαση μέσω `PV_THESIS_DATA_DIR`, redistribution **REVIEW REQUIRED**.
- ETSIDI raw electrical/weather: δεν αντιγράφονται στο public-ready tree, redistribution **REVIEW REQUIRED**.
- PVGIS/TMY frozen responses: διατηρούνται με provider/provenance CSV και hashes, redistribution **REVIEW REQUIRED** πριν από publication.

Τα ακριβή imported artifacts και SHA-256 καταγράφονται στα `PROVENANCE.json` και `CONSOLIDATED_BRANCH_IMPORT.json`. Κανένα path του CLI δεν απαιτεί προσωπικό Windows username.
