from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .doctor import run_doctor
from .branches import BRANCHES, reproduce_frozen_branches, verify_branches
from .execution import reproduce_canonical, reproduce_raw
from .frozen import reproduce_frozen
from .paths import project_root
from .reporting import generate_reports, generate_software_reports
from .smoke import run_smoke
from .verification import run_verification


def _print(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=lambda value: value.item() if hasattr(value, "item") else str(value)))


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(prog="pv-thesis", description="Αναπαραγωγή πρόβλεψης της διπλωματικής Φ/Β παραγωγής")
    command.add_argument("--root", help="Project root. Εναλλακτικά ορίστε PV_THESIS_HOME.")
    top = command.add_subparsers(dest="command", required=True)
    top.add_parser("doctor", help="Έλεγχος Python, εξαρτήσεων, hashes και ιδιωτικών δεδομένων")
    verify_all = top.add_parser("verify", help="Επαλήθευση frozen artifacts ενός ή όλων των επιστημονικών κλάδων")
    verify_group = verify_all.add_mutually_exclusive_group(required=True)
    verify_group.add_argument("--all", action="store_true")
    verify_group.add_argument("--branch", choices=BRANCHES)
    reproduce_all = top.add_parser("reproduce", help="Επανυπολογισμός μετρικών από frozen outputs, χωρίς training")
    reproduce_group = reproduce_all.add_mutually_exclusive_group(required=True)
    reproduce_group.add_argument("--all", action="store_true")
    reproduce_group.add_argument("--branch", choices=BRANCHES)
    reproduce_all.add_argument("--mode", choices=["frozen"], required=True)
    top.add_parser("report", help="Παραγωγή τελικών αναφορών ενοποιημένου λογισμικού")
    forecasting = top.add_parser("forecasting", help="Επαλήθευση και αναπαραγωγή πρόβλεψης")
    sub = forecasting.add_subparsers(dest="forecasting_command", required=True)
    sub.add_parser("verify", help="Έλεγχος παγωμένου πρωτοκόλλου και artifacts")
    smoke = sub.add_parser("smoke", help="Πραγματικό Stage D smoke training ανά οικογένεια")
    smoke.add_argument("--output")
    reproduce = sub.add_parser("reproduce", help="Εκτέλεση frozen, canonical ή raw mode")
    reproduce.add_argument("--mode", choices=["frozen", "canonical", "raw"], required=True)
    reproduce.add_argument("--output")
    reproduce.add_argument("--confirm-full-training", action="store_true", help="Ρητή επιβεβαίωση για πολυώρη canonical/raw εκπαίδευση")
    sub.add_parser("report", help="Παραγωγή δομημένων τελικών αναφορών από τα διαθέσιμα τεκμήρια")
    return command


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    root = project_root(args.root)
    if args.command == "doctor":
        result = run_doctor(root)
    elif args.command == "verify":
        result = verify_branches(root, list(BRANCHES) if args.all else [args.branch])
    elif args.command == "reproduce":
        result = reproduce_frozen_branches(root, list(BRANCHES) if args.all else [args.branch])
    elif args.command == "report":
        result = generate_software_reports(root)
    elif args.forecasting_command == "verify":
        result = run_verification(root)
    elif args.forecasting_command == "smoke":
        result = run_smoke(root, args.output)
    elif args.forecasting_command == "report":
        result = generate_reports(root)
    elif args.mode == "frozen":
        result = reproduce_frozen(root, args.output)
    elif args.mode == "canonical":
        result = reproduce_canonical(root, args.output, args.confirm_full_training)
    else:
        result = reproduce_raw(root, args.output, args.confirm_full_training)
    _print(result)
    return 0 if result.get("status") == "PASS" else (3 if result.get("status") in {"BLOCKED", "NOT TESTED"} else 1)
