# Copyright (c) 2024, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
"""Run TIGER training without torchrun, with an in-process shim for the
broken `dynamicemb_util` module.

This is a smoke-test launcher only — for normal use, prefer
`ft_scripts/train_tiger.sh` (which uses torchrun). This launcher exists
so we can run the integration end-to-end on machines where:

    (a) the local source's `tzrec.utils.dynamicemb_util` imports a
        symbol from torchrec that's not present in the installed version
        (`HardwarePerfConfig`), and
    (b) GPU isn't available (old CUDA driver) — falls back to gloo/CPU.

Usage:
    python ft_scripts/tiger_train_launcher.py \\
        --pipeline_config_path ft_scripts/tiger.config \\
        --train_input_path 'data/tiger_synth_train/*.parquet' \\
        --eval_input_path  'data/tiger_synth_eval/*.parquet' \\
        --model_dir experiments/tiger_smoke
"""

import argparse
import os
import sys
import types

# ---------------------------------------------------------------------------
# Step 0: ensure local source on sys.path wins over an installed tzrec.
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ---------------------------------------------------------------------------
# Step 1: install the dynamicemb shim BEFORE any tzrec import.
# ---------------------------------------------------------------------------
_shim = types.ModuleType("tzrec.utils.dynamicemb_util")
_shim.__spec__ = None  # pyre-ignore
_shim.has_dynamicemb = False
_shim.is_dynamicemb_enabled = lambda: False
_shim.apply_dynamicemb_patches = lambda *a, **k: None
_shim.build_dynamicemb_constraints = lambda *a, **k: None
_shim.dynamicemb_calculate_shard_storages = lambda *a, **k: None
sys.modules["tzrec.utils.dynamicemb_util"] = _shim


# ---------------------------------------------------------------------------
# Step 2: set distributed env vars so single-process CPU runs work
# without `torchrun` wrapping us.
# ---------------------------------------------------------------------------
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "32988")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline_config_path", required=True)
    parser.add_argument("--train_input_path", default=None)
    parser.add_argument("--eval_input_path", default=None)
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--edit_config_json", default=None)
    args = parser.parse_args()

    # Now safe to import tzrec.
    from tzrec.main import train_and_evaluate  # noqa: E402

    train_and_evaluate(
        pipeline_config_path=args.pipeline_config_path,
        train_input_path=args.train_input_path,
        eval_input_path=args.eval_input_path,
        model_dir=args.model_dir,
        edit_config_json=args.edit_config_json,
    )


if __name__ == "__main__":
    main()
