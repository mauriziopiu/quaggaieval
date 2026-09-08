"""
run.py — PyInstaller entry-point shim
======================================
PyInstaller requires a plain .py script as its Analysis entry-point;
it cannot use `python -m quaggaieval_desktop` directly.

This file lives at the project root alongside the quaggaieval_desktop/ package.
It is NOT used during normal development — use `python -m quaggaieval_desktop` instead.
"""
from quaggaieval_desktop.__main__ import main

if __name__ == '__main__':
    main()