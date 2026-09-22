#!/usr/bin/env python
"""Near-circular study, n = 30 extension.

    python scripts/run_near_circular_n30.py --arm cold
    python scripts/run_near_circular_n30.py --arm transfer
    python scripts/run_near_circular_n30.py --report
    python scripts/run_near_circular_n30.py --arm cold transfer --report \
           --results-dir /path/to/results

Equivalent to the ``gravinns-nc30`` command installed with the package.
"""
import matplotlib

matplotlib.use("Agg")   # headless: panels are saved as PNGs, nothing is shown

from gravinns.experiments.near_circular_n30 import main

if __name__ == "__main__":
    main()
