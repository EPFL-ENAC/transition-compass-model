# TransitionCompassModel — Multi-Sector Climate Pathway Calculation Engine

Python library that computes emission, energy, and environmental indicator pathways sector by sector, given policy lever settings. Powers the [TransitionCompassViz](https://github.com/EPFL-ENAC/leure-speed-to-zero) visualization platform.

## Live Platforms

- **Production**: [https://transition-compass.epfl.ch/](https://transition-compass.epfl.ch/)
- **Development**: [https://transition-compass-dev.epfl.ch/](https://transition-compass-dev.epfl.ch/)

## Dual-Repository Architecture

This project is split into two repositories:

| Repository | Purpose |
|---|---|
| **[speed-to-zero](https://github.com/EPFL-ENAC/leure-speed-to-zero)** | Web application — frontend UI, backend API, deployment |
| **[transition-compass-model](https://github.com/2050Calculators/transition-compass-model)** (this repo) | Climate calculation engine — sector modules, data matrices, optimization |

- **Model researchers** (sector calculations, data, parameters): work here
- **App developers** (frontend, API, charts, UI): work in [speed-to-zero](https://github.com/EPFL-ENAC/leure-speed-to-zero)

## Architecture

The model is structured around independent sector modules coordinated by an interaction runner:

```
transition_compass_model/
├── model/
│   ├── interactions.py          # Runs all sectors and resolves cross-sector dependencies
│   ├── agriculture_module.py
│   ├── ammonia_module.py
│   ├── buildings_module.py
│   ├── buildings/               # Sub-modules for buildings sector
│   ├── climate_module.py
│   ├── district_heating_module.py
│   ├── emissions_module.py
│   ├── energy_module.py         # EnergyScope LP (Pyomo/AMPL)
│   ├── energy/                  # Sub-modules for energy sector
│   ├── forestry_module.py
│   ├── industry_module.py
│   ├── landuse_module.py
│   ├── lca_module.py
│   ├── lifestyles_module.py
│   ├── minerals_module.py
│   ├── oilrefinery_module.py
│   ├── power_module.py
│   ├── transport_module.py
│   └── transport/               # Sub-modules for transport sector
├── config/
│   └── lever_position.json      # Default lever settings (1–4 ambition levels)
└── _database/
    └── data/
        └── datamatrix/          # Regional data matrices (Vaud, Switzerland, EU27)
```

**Sectors**: Agriculture, Ammonia, Buildings, Climate, District Heating, Emissions, Energy (LP), Forestry, Industry, Land Use, LCA, Lifestyles, Minerals, Oil Refinery, Power, Transport

Given a set of policy lever values (integers 1–4 representing ambition levels), `interactions.py` runs the relevant sector modules and returns time-series results for emissions, energy demand, and environmental indicators.

## Git LFS

This repository uses [Git LFS](https://git-lfs.com/) to store binary data files (`.pickle`, `.pdf`) on GitHub's built-in LFS storage. You need Git LFS installed for a working clone:

```bash
# Install Git LFS (once per machine)
git lfs install
```

After cloning, LFS files are fetched automatically. If you see small pointer files instead of real data, run `git lfs pull`.

## Install

> **Note on Git LFS**: `pip install git+https://...` does **not** support Git LFS — it will install pointer files instead of real data. To use this package:
>
> - **Development / research** — install from a local clone (recommended):
>   ```bash
>   git clone https://github.com/2050Calculators/transition-compass-model.git
>   cd transition-compass-model
>   pip install .
>   ```
> - **As a `uv` dependency** (e.g. in another project or app) — works if [`git lfs install`](https://git-lfs.com/) has been run on the machine. This is how [speed-to-zero](https://github.com/EPFL-ENAC/leure-speed-to-zero) consumes the model in production.

Or pin to a specific release tag:

```bash
git clone --branch v1.2.3 https://github.com/2050Calculators/transition-compass-model.git
cd transition-compass-model
pip install .
```

## Usage

```python
from transition_compass_model.model.common.config_loader import load_lever_config
from transition_compass_model.model.common.auxiliary_functions import (
    filter_country_and_load_data_from_pickles,
)
from transition_compass_model.model.interactions import runner
import logging

logger = logging.getLogger(__name__)

lever_setting = load_lever_config()
years_setting = [1990, 2023, 2025, 2050, 5]  # [start_ots, end_ots, start_fts, end_fts, fts_step]
country_list = ["Switzerland", "EU27", "Vaud"]
sectors = ["climate", "lifestyles", "buildings", "transport", "industry",
           "forestry", "agriculture", "ammonia", "lca"]

DM_input = filter_country_and_load_data_from_pickles(
    country_list=country_list, modules_list=sectors
)

output, KPI = runner(lever_setting, years_setting, DM_input, sectors, logger)
```

To customise lever settings, edit `transition_compass_model/config/lever_position.json` (each value is an integer 1–4).

## Local Development

### Standalone (model only)

```bash
git lfs install  # If not already done
git clone https://github.com/2050Calculators/transition-compass-model.git
cd transition-compass-model
make install     # Install deps + activate pre-commit hooks
```

| Command | Description |
|---|---|
| `make install` | Install all dependencies (incl. dev) and register git pre-commit hooks |
| `make format` | Auto-format code with ruff |
| `make lint` | Check code quality with ruff |
| `make test` | Run test suite with pytest |

Run the full model:

```bash
uv run python -m transition_compass_model.model.interactions_localrun
```

Run individual sector modules:

```bash
uv run python -m transition_compass_model.model.transport_module
uv run python -m transition_compass_model.model.buildings_module
# etc.
```

### With the app (see model changes in the UI)

Clone both repos as siblings and use the app's local install:

```
parent-dir/
├── speed-to-zero/                  ← app repo
└── transition-compass-model/       ← this repo
```

```bash
cd speed-to-zero
make install-dev   # Installs deps + editable model from sibling folder
make run           # Start app — model changes are reflected immediately
```

Verify which model is active:

```bash
# From speed-to-zero/
cd backend && make check-model
# Local mode:  path points to ../../transition-compass-model/
# Remote mode: path points inside .venv/lib/.../site-packages/
```

See [DEVELOPMENT.md](DEVELOPMENT.md) for full details (IDE setup, switching between local/remote model, etc.) and [CONTRIBUTING.md](CONTRIBUTING.md) for the contribution workflow (branching, commits, pull requests).

## Versioning and Releases

Releases follow [Semantic Versioning](https://semver.org/): `vMAJOR.MINOR.PATCH`

| Part | When to bump |
|---|---|
| `MAJOR` | Breaking changes to the module API or data format |
| `MINOR` | New sectors, modules, or significant new features |
| `PATCH` | Bug fixes, parameter tweaks, data updates |

### Release flow

1. Commit and push all model changes to `main`
2. Tag and push: `git tag v1.2.3 && git push origin v1.2.3`
3. GitHub Actions dispatches a `model-updated` event to the app repo
4. A bump PR is created automatically in speed-to-zero (`chore/bump-model-v1.2.3 → dev`)
5. Review and merge the PR — CI validates the build, then deploys

## License

[Apache License 2.0](LICENSE)
