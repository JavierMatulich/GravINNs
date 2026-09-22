#!/usr/bin/env python
"""Long-orbit case (N = 50 orbits, five ICs).

    python scripts/run_long_orbit.py --ic e006 e019 e036 e051 e064 --refine adam
    python scripts/run_long_orbit.py --show

Equivalent to the ``gravinns-longorbit`` command installed with the package.
"""
import matplotlib

matplotlib.use("Agg")

from gravinns.experiments.long_orbit import main

if __name__ == "__main__":
    main()
