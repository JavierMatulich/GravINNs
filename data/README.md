# data/

Reference-orbit databases (`.npz`) used by the parametric case
(`notebooks/03_parametric.ipynb`). All are written by
`gravinns.parametric.build_reference_cache` and read by
`load_reference_cache` / `load_reference_cache_lazy`.

| File | Used for | Size | In the repository? |
|---|---|---|---|
| `refs_20000_10orb.npz` | **anchor orbits** for every trained model in the paper (section 1 of the notebook) | large | to be added (see below) |
| `refs_10orb_test.npz` | 100 000 test orbits for sections 3 (evaluation) and 4 (paper statistics) | ~3 GB | no — generate with section 2 of the notebook |

All databases: 2PN, `n_orbits = 10`, `n_pts = 3000`.

## Adding `refs_20000_10orb.npz`

GitHub rejects files larger than 100 MB in a normal commit. If the file is
larger than that, use [Git LFS](https://git-lfs.com):

```bash
git lfs install
git lfs track "data/refs_20000_10orb.npz"
git add .gitattributes data/refs_20000_10orb.npz
git commit -m "Add anchor-orbit database"
```

or host it elsewhere (e.g. Zenodo) and put the download link here.

`.gitignore` ignores `*.npz` everywhere except `data/refs_20000_10orb.npz`,
so the 3 GB test databases are never committed by accident.
