# Changelog

## [1.3] - 2026-05-04
First public release since v1.0. Adds an EIT helpers module (protocols,
Fisher information, D-optimal block selection), expands the optimizer with
high-level loss/Jacobian/Hessian wrappers and a prior-centered LM step,
fixes a real bug in the analytic Jacobian that was missing the
outermost-tissue dEIT contribution, adds finite-difference regression
tests for both Jacobian and Hessian, and modernizes packaging. (Internal
versions 1.1 and 1.2 were not tagged or published.)

### Added
- New module `pyreite/EIThelpers.py`:
  - `EIT_protocol`, `apply_protocol`, `build_full_protocol` for measurement
    protocol construction.
  - `compute_fim_from_J`, `compute_fim_from_blocks` for Fisher information
    matrices in log-conductivity space.
  - `d_optimality`, `info_metric` (`min_eig` / `logdet`) for design criteria.
  - `greedy_select_blocks_doptimal`, `greedy_select_until_full` for greedy
    D-optimal pattern selection.
  - `scale_experiment_to_simulation` for matching experiment/simulation
    moments.
- New solvers in `pyreite/optimizers.py`:
  - `levenberg_marquardt_hessiancheck_prior_centered` — Tikhonov-regularized
    LM with optional Hessian acceleration and prior-centered shrinkage.
  - `build_Lpr0_from_J`, `build_Lpr0_combined` — Minpack-style damping
    scaling (sensitivity-only and sensitivity + prior-precision combined).
  - `_as_reg_matrix` Lpr0 vector-or-matrix accessor.
- `tests/conftest.py` with autouse RNG seeding (numpy + stdlib `random`)
  for reproducible test runs.
- `tests/test_material_derivative.py::test_jacobian_finite_difference`
  parametrized over 1, 2, and 3 meshes — analytic vs. central-difference
  comparison at near-machine precision.
- `tests/test_material_derivative.py::test_hessian_finite_difference`
  parametrized over 2 and 3 meshes.
- `tests/test_EIThelpers.py` — 17 tests covering the new helpers
  (protocols, FIM, D-optimality, greedy selection, scaling).
- `tests/test_optimizers.py` — 9 new tests for `tikhonov`, prior-centered
  LM, NOSER LM, and `build_Lpr0_*` damping helpers.
- `pyreite/colors.py` — single source for ANSI helpers, replacing
  duplicate `bcolors`/`printred`/etc. blocks in `optimizers.py` and
  `examples/cond_colin.py`.
- `setup.py` now declares `python_requires='>=3.7'` and
  `install_requires=['numpy>=1.21.6', 'h5py>=3.8.0', 'scipy>=1.7.3']`.
- Project logo (`logo.png`).

### Changed
- **API**: `material_derivative.jacobian`, `material_derivative.hessian`,
  `optimizers.loss_residuals`, `optimizers.jac`, `optimizers.hess`, and
  `optimizers.jac_hess` gained `ND2V=None, protocol=None` keyword
  arguments for measurement selection.
- **Move**: `EIT_protocol` moved from `pyreite.material_derivative` to
  `pyreite.EIThelpers`. Update imports accordingly.
- CI matrix expanded from Python 3.6 to 3.7–3.11; container image is
  now per-version (`ghcr.io/harmening/pyreite:py3.${{ matrix.python_version }}`).
- Dependency upgrades: numpy 1.25, OpenMEEG 2.4.7.
- `requirements.txt` lower-bounds pinned: `numpy>=1.21.6`, `h5py>=3.8.0`,
  `scipy>=1.7.3`.
- README install command updated from deprecated
  `python setup.py install` to `pip install .`.
- Dockerfile streamlined (39 → 7 lines).
- `examples/cond_colin.py` rewritten to use the new high-level optimizer
  wrappers and protocol selection.
- Project slogan: **Pythonic, Yet Rudimentary, EIT Expert** →
  **Pythonic, Yet Robust, EIT Expert** (PYREITE acronym preserved).
  Updated in README and `setup.py` description.

### Fixed
- Analytic Jacobian outermost-tissue column was missing the dEIT
  contribution because it indexed an empty `ind['p'][-1]` block; the
  same bug propagated into the Hessian via `d2EIT`. Now caught by
  finite-difference regression tests.
- `OpenMEEGHead.set_cond` no longer keeps stale `first_derivatives`,
  `_eitsm`, `_gain`, `_C`, `_V`, or `_h2em` after a conductivity update.
- `OpenMEEGHead.__init__` and `set_cond` use a `tempfile.TemporaryDirectory`
  context manager — no more orphaned `/tmp/*.{geom,cond,elec,tri}` files
  on Python exceptions; no more random-name collisions across processes.
- `OpenMEEGHead.Vsetter` no longer silently ignores caller-supplied
  `freqs`/`Iamp`/`ref`/`excluded_chan` when `_V` was already cached.
- `tests/test_geometry.py` no longer leaks `tmp_test.{cond,geom}` /
  `tmp_tri*.tri` into the repo root; both tests use
  `tempfile.TemporaryDirectory`.



## [1.0] - 2023-09-15
Adds the analytic conductivity Jacobian and Hessian for the OpenMEEG
symmetric BEM head model, plus a Levenberg–Marquardt optimizer suite for
recovering tissue conductivities from EIT measurements. Builds on the
v0.2 head model to deliver an end-to-end EIT inverse-solver pipeline.

### Added
- `pyreite/material_derivative.py` — first analytic Jacobian/Hessian
  derivation for the symmetric BEM system matrix w.r.t. tissue
  conductivities:
  - `first_derivatives`, `second_derivatives` — closed-form derivatives
    of the system matrix.
  - `jacobian_per_measurements`, `jacobian` and `hessian_per_measurement`,
    `hessian` — per-measurement and per-tissue derivative builders.
  - `dAds1`–`dAds4` and `dAds1ds1`–`dAds4ds4` — per-tissue derivative
    blocks for 1- to 4-shell heads.
  - `EIT_protocol` — measurement-protocol mask builder
    (`'all'` / `'all_realistic'` / explicit list).
- `pyreite/optimizers.py` — log-space Levenberg–Marquardt suite for
  conductivity inversion:
  - `loss_residuals`, `jac`, `hess`, `jac_hess` — high-level wrappers
    that drive the head model through one optimizer step.
  - `tikhonov`, `levenberg_marquardt_hessian`,
    `levenberg_marquardt_hessiancheck`, `levenberg_marquardt_hessian_noser`
    — solver variants with Tikhonov regularization, Hessian acceleration,
    posdef filtering, and NOSER-weighted regularization.
  - `is_posdef` PSD check, ANSI helpers (`printred`/`printyellow`/…) for
    iteration logging.
- `tests/test_material_derivative.py`, `tests/test_optimizers.py`.
- `examples/cond_colin.py` — end-to-end example: simulate EIT on the
  Colin27 head, perturb conductivities, recover them via LM with
  Hessian acceleration.
- Smaller `tests/test_data/` meshes (cortex/csf/scalp/skull `.tri`) for
  faster CI runs.



## [0.2] - 2023-04-26
Initial public release. Provides EIT forward-model simulation on a
nested-shell head model via the OpenMEEG BEM solver, plus the geometric
plumbing (mesh I/O, electrode-to-scalp projection, OpenMEEG file
generation) needed to drive it. Tested with Python 3.6 and OpenMEEG 2.4.0.

### Added
- `pyreite/OpenMEEGHead.py` — `OpenMEEGHead` class wrapping the OpenMEEG
  symmetric BEM with NumPy I/O. Exposes lazily-evaluated cached
  properties for the system matrix `A`, its inverse `Ainv`, the EIT
  source matrix `eitsm`, head-to-electrode operator `h2em`, the gain
  matrix `gain`, and the EIT voltage tensor `V` (with optional reference,
  current-amplitude scaling, NaN-masking, and excluded-channel handling).
- `pyreite/data_io.py` — readers/writers for `.tri` mesh files and
  OpenMEEG `.geom`, `.cond`, `.elec`, `.dip` files; surface-normal
  estimation (`get_normals`, `normals_for_faces`, `vertex_normals`).
- `pyreite/geometry.py` — `create_geometry` (build OpenMEEG `Geometry` +
  `Sensors`), `mesh2bnd`, plus a port of FieldTrip's `align_electrodes`
  (project-to-scalp via `ptriprojn` / `lmoutrn` / `plinprojn` /
  `routlm`).
- `tests/test_OpenMEEGHead.py`, `tests/test_data_io.py`,
  `tests/test_geometry.py`, plus `tests/data_for_testing.py` with an
  icosphere generator (`form_base_icosahedron` + `subdivision`) for
  synthetic nested-shell test cases.
- GitHub Actions CI workflow (`.github/workflows/action.yml`) running
  the test suite in containerized Python 3.6.

