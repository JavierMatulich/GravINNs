"""Parametric angle-PINN: ONE network N(phi, theta) for a 4-parameter box of
initial conditions theta = (p0_factor, r0_km, pr0_factor, nu).

Origin: GitHub_parametric.ipynb.  Everything the notebook calls is importable
from here::

    from gravinns.parametric import *

Modules
    orbits       PN Hamiltonian (shared), RK reference orbits, parametric residual
    model        ParametricAnglePINN
    training     train_parametric_pinn, make_anchor_thetas, make_probe_thetas
    checkpoints  load_parametric_model, get_param_ranges
    refine       refine_parametric_pinn
    analysis     l2_sweep_1d
    long_orbit   generate_long_orbit, generate_long_orbit_section, OrbitClock
    benchmark    reference caches, eval_model_on_cache, heatmaps, error filter
    diagnostics  eccentricity-vs-periapsis tests, memory-safe helpers
    paper_stats  Tables IV-VI: score_once, joint_table, compare_models, ...
    workflow     notebook helpers (ref data from cache, predict_orbit, ecc bins)
"""
from .analysis import l2_sweep_1d
from .benchmark import (ErrorFilterNet, build_benchmark_db, build_error_dataset,
                        build_reference_cache, compare_models_on_cache,
                        db_pair_heatmaps, eval_chunked, eval_model_on_cache,
                        load_benchmark_db, load_reference_cache,
                        load_reference_cache_lazy, rk4_orbits_batch,
                        score_db, score_on_cache, train_error_filter)
from .checkpoints import get_param_ranges, load_parametric_model
from .diagnostics import (is_lazy, materialise, memory_report, omega_accuracy,
                          subsample_cache)
from .long_orbit import OrbitClock, generate_long_orbit, generate_long_orbit_section
from .model import ParametricAnglePINN
from .orbits import C_QD, C_SQ, build_reference_set, integrate_orbit, phi_residual_param
from .paper_stats import (cache_strata, compare_models, compare_models_joint,
                          joint_table, joint_table_2d, periapsis_rg, score_once,
                          variance_decomposition)
from .refine import refine_parametric_pinn
from .training import make_anchor_thetas, make_probe_thetas, train_parametric_pinn
from .workflow import (anchors_and_probes_from_cache, build_ref_data_from_cache,
                       ecc_bin_boxplot, ecc_bin_table, predict_orbit)

__all__ = [n for n in dir() if not n.startswith("_")
           and n not in ("analysis", "benchmark", "checkpoints", "diagnostics",
                         "long_orbit", "model", "orbits", "paper_stats",
                         "refine", "training", "workflow")]
