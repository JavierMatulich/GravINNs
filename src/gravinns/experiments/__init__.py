"""One module per experiment / paper case.

Each module holds only what is *specific* to that case (configuration,
initial-condition selection, output folders, report tables) and imports the
shared machinery from the rest of the package.

    near_circular        shared config of the near-circular study (COMMON_NC, ARMS_NC)
    near_circular_n30    extend every stratum to n = 30  ->  run_to30(), report30()
    long_orbit           N = 50 orbits, 5-IC eccentricity ladder  ->  run_ic(), show_results()
    dissipative          2.5PN inspiral survey (24 runs)  ->  run_case(), paper_table()
"""
