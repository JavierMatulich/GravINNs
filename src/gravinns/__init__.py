"""GravINNs -- physics-informed neural networks for post-Newtonian binary orbits.

Sub-packages
    constants    physical set-up and code-unit normalisation
    physics      PN Hamiltonian (0-3PN), 2.5PN radiation reaction
    models       angle-parametrised orbit PINN
    training     near-circular, long-orbit, and dissipative protocols
    recording    on-disk plot record, optional loss-component log
    plotting     figure helpers
    sweep        IC generation, sweep driver, offline analysis
    parametric   parametric PINN N(phi, theta) over a box of initial conditions
    experiments  one module per study / paper case
"""
__version__ = "1.0.1"
