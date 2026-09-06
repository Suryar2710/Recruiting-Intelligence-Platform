# Engineering Notes

## Canonical environment & reproducibility

Data generation and model training are standardized on the Docker/CI environment
(Python 3.11, numpy 1.26.4) because numpy major-version differences were found to
produce different simulated outputs from the same fixed random seed. Host-machine
runs on other Python/numpy versions are not guaranteed to reproduce identical
results.

### Why this happens
The synthetic data generator (`phase1_generate_data/generate.py`) uses a fixed
seed (`SEED=42`). A seed guarantees the same underlying random bit-stream, **not**
the same emitted values across numpy major versions — numpy 2.x changed how some
RNG calls consume that stream, so the same seed yields a different dataset under
numpy 2.x (host, Python 3.14) than under numpy 1.26.4 (Docker/CI, Python 3.11).

### The decision
- **Canonical dataset**: the one produced under Python 3.11 + numpy 1.26.4, which
  has **10,836 hires** across 120,000 applications. Any reference to 7,957 hires
  is stale (it came from a host run) and is not used anywhere in the current repo.
- `numpy==1.26.4` is **pinned exactly** in `requirements.txt` to prevent a future
  `pip install` from silently drifting to numpy 2.x.
- CI (GitHub Actions, Python 3.11) regenerates the dataset at full scale on every
  push and **asserts the hire count is exactly 10,836** — if a dependency ever
  drifts, the build fails loudly rather than silently producing different numbers.

### What this means for the numbers in this repo
All model-card metrics, KPI figures, and validated signal gaps come from the
canonical environment. `docker compose up --build` reproduces them; a fresh CI run
reproduces them; a host run on a different numpy version may not.
