# Repository Guidelines

## Project Structure & Module Organization

TorchEasyRec is a Python recommendation framework packaged as `tzrec`. Core source lives in `tzrec/`, with model implementations in `tzrec/models/`, reusable layers in `tzrec/modules/`, datasets in `tzrec/datasets/`, features in `tzrec/features/`, generated protobuf code under `tzrec/protos/`, and utilities in `tzrec/utils/` and `tzrec/tools/`. Tests are mostly colocated as `*_test.py` files under `tzrec/`, with integration runners and mock configs in `tzrec/tests/`. Example training configs are in `examples/`, docs in `docs/`, CI helpers in `scripts/ci/`, and small test assets in `data/test/`.

## Build, Test, and Development Commands

- `pip install -r requirements.txt`: install the full development dependency set.
- `pre-commit install`: enable repository formatting and validation hooks.
- `bash scripts/gen_proto.sh`: regenerate protobuf Python files before tests or packaging.
- `bash scripts/ci/ci_test.sh`: install GPU/extra requirements, prepare CI data, and run the main unittest suite through `tzrec/tests/run.py`.
- `MKL_THREADING_LAYER=GNU PYTHONPATH=. python tzrec/tests/run.py`: run discovered unit tests without reinstalling dependencies.
- `python -m tzrec.models.deepfm_test DeepFMTest.test_name`: run one unittest case.
- `python scripts/pyre_check.py`: run strict Pyre type checks.
- `bash scripts/build_wheel.sh nightly` or `release`: build source and wheel distributions.

## Coding Style & Naming Conventions

Use PEP 8 Python style with 4-space indentation. Formatting and linting are enforced by `ruff` and `ruff-format`; configuration is in `.ruff.toml`. Docstrings follow Google style where required. Test files use the `*_test.py` suffix, unittest classes typically use `CamelCaseTest`, and test methods use `test_*`. Preserve existing module naming: lowercase Python modules, descriptive model names matching config/model classes, and generated protobuf files only from `scripts/gen_proto.sh`.

## Testing Guidelines

Tests use Python `unittest`, with `hypothesis` and `parameterized` available for data-driven coverage. Add focused tests near the changed module when possible, and add or update `tzrec/tests/configs/*.config` for integration behavior. Run the narrow test first, then `MKL_THREADING_LAYER=GNU PYTHONPATH=. python tzrec/tests/run.py` or `bash scripts/ci/ci_test.sh` before submitting larger changes.

## Commit & Pull Request Guidelines

Recent commits use bracketed prefixes such as `[feat]`, `[bugfix]`, `[ci]`, `[doc]`, and `[chore]`, followed by a concise imperative summary and often a PR number, for example `[bugfix] Kafka: batch + retry offsets_for_times ... (#547)`. Pull requests should describe the behavioral change, list validation commands run, link related issues, and include docs or config updates when user-facing behavior changes.

## Security & Configuration Tips

Do not commit credentials, Alibaba Cloud tokens, OSS credentials, or private dataset paths. Keep large generated artifacts, caches, and local training outputs out of the repository unless they are intentional small fixtures under `data/test/`.
