# Copyright (c) 2024, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Measure the GenRec training step under each attention kernel.

PR #663 packs the teacher-forced forward through the FlashAttention-2 varlen
kernel instead of padding every row to the longest row in its batch, and states
that the speedup is unmeasured. This drives the real ``GenRecCausalLMModel`` over
real prompt assembly and real sample rows, so the number reflects the model that
trains rather than a stand-in backbone.

The gain tracks how ragged the length distribution is, so the padding waste of
each batch is reported beside the timing.

Example:
    PYTHONPATH=. python -m tzrec.tools.genrec_attn_kernel_bench \
        --backbone /path/to/Qwen2.5-0.5B \
        --data-glob '/path/to/samples/*.parquet' \
        --batch-sizes 20,40
"""

import argparse
import glob
import json
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq
import torch
from google.protobuf import text_format

from tzrec.datasets.utils import Batch
from tzrec.features.feature import FgMode, create_features
from tzrec.main import _create_model
from tzrec.prompt.assembler import OUTPUT_KEYS, PromptAssembler
from tzrec.prompt.compile import compile_prompt
from tzrec.protos import feature_pb2
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.protos.models.genrec_model_pb2 import GenRecModelConfig
from tzrec.protos.prompt_pb2 import PromptConfig

_MIB = 2**20


def _build_model(
    args: argparse.Namespace, attn_kernel: int, device: torch.device
) -> Tuple[torch.nn.Module, object]:
    """Build the real GenRec model for one attention kernel.

    Args:
        args: parsed command line, carrying the prompt and backbone description.
        attn_kernel: ``GenRecModelConfig.AttnKernel`` value.
        device: where the model is placed.

    Returns:
        Tuple[torch.nn.Module, object]: the model and the compiled prompt it
            was built on, which the batch builder needs.
    """
    feature_config = feature_pb2.FeatureConfig()
    text_format.Merge(
        f'sequence_raw_feature {{ feature_name: "{args.hist_feature}" '
        f'expression: "user:{args.hist_feature}" value_dim: 1 '
        f"sequence_length: {args.sequence_length} }}",
        feature_config,
    )
    features = create_features([feature_config], fg_mode=FgMode.FG_NONE)

    prompt_config = PromptConfig(
        tokenizer_path=args.tokenizer or f"{args.backbone}/tokenizer.json",
        prompt=args.prompt.replace("{{HIST}}", "{{" + args.hist_feature + "}}"),
        response="{{" + args.label_field + "}}",
        max_length=args.max_length,
    )
    prompt_config.sid_space.codebook.extend([int(x) for x in args.codebook.split(",")])
    prompt_config.sid_space.vocab_pad_to_multiple_of = args.vocab_pad_to_multiple_of
    compiled_prompt = compile_prompt(prompt_config, features, [args.label_field])

    model_config = ModelConfig()
    lm_config = model_config.genrec_causal_lm_model
    lm_config.hf_model_name_or_path = args.backbone
    lm_config.common.beam_widths.extend([int(x) for x in args.beam_widths.split(",")])
    lm_config.common.num_return_sequences = args.num_return_sequences
    lm_config.common.ignore_index = -100
    lm_config.common.lm_parameter_dtype = GenRecModelConfig.ParamDtype.Value(args.dtype)
    lm_config.common.attn_kernel = attn_kernel

    # same draw for every kernel, so the reported losses are comparable
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(args.seed)
        model = _create_model(
            model_config, features, [args.label_field], compiled_prompt=compiled_prompt
        )
    return model.to(device), compiled_prompt


def _build_batch(
    args: argparse.Namespace,
    compiled_prompt: object,
    rows: Tuple[List[List[int]], List[List[int]]],
    device: torch.device,
) -> Batch:
    """Assemble one training batch through the real ``PromptAssembler``.

    Args:
        args: parsed command line, naming the history and label columns.
        compiled_prompt: the compiled prompt the model was built on.
        rows: per-row history and answer codes, already offset by level.
        device: where the batch is placed.

    Returns:
        Batch: what ``model.predict`` consumes.
    """
    hist_rows, answer_rows = rows
    parsed = {
        f"{args.hist_feature}.values": torch.tensor(
            [code for row in hist_rows for code in row], dtype=torch.float32
        ).reshape(-1, 1),
        f"{args.hist_feature}.lengths": torch.tensor([len(r) for r in hist_rows]),
        f"{args.label_field}.values": torch.tensor(
            [code for row in answer_rows for code in row], dtype=torch.int64
        ),
        f"{args.label_field}.lengths": torch.tensor([len(r) for r in answer_rows]),
    }
    streams = PromptAssembler(
        compiled_prompt.prompt_plan,
        compiled_prompt.sid_space,
        include_response=True,
    )(parsed)
    batch = Batch()
    batch.additional_infos.update({key: streams[key] for key in OUTPUT_KEYS})
    return batch.to(device)


def _read_rows(
    args: argparse.Namespace, batch_size: int, seed: int
) -> Tuple[List[List[int]], List[List[int]]]:
    """Draw real sample rows from the parquet shards.

    Args:
        args: parsed command line, carrying the data glob and column names.
        batch_size: how many rows to draw.
        seed: selection seed, so every kernel sees identical rows.

    Returns:
        Tuple[list, list]: history rows and answer rows of offset SID codes.
    """
    paths = sorted(glob.glob(args.data_glob))
    if not paths:
        raise ValueError(f"no parquet matched {args.data_glob}")
    table = pq.read_table(paths[0], columns=[args.hist_feature, args.label_field])
    index = np.random.default_rng(seed).choice(
        table.num_rows, size=batch_size, replace=False
    )
    table = table.take(index)
    return (
        table.column(args.hist_feature).to_pylist(),
        table.column(args.label_field).to_pylist(),
    )


def _time_steps(
    model: torch.nn.Module,
    batch: Batch,
    iters: int,
    warmup: int,
    autocast: bool,
) -> Tuple[float, float, float]:
    """Run train steps and return latency, peak memory and the last loss.

    Args:
        model: the GenRec model, already on device.
        batch: the assembled batch to repeat.
        iters: timed iterations.
        warmup: untimed iterations before timing starts.
        autocast: wrap the forward in bfloat16 autocast, as mixed_precision does.

    Returns:
        Tuple[float, float, float]: ms per step, peak MiB, final loss value.
    """
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    def step() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            predictions = model.predict(batch)
            loss = model.loss(predictions, batch)["ce_loss"]
        loss.backward()
        optimizer.step()
        return loss.detach()

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(iters):
        loss = step()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000 / iters
    return elapsed, torch.cuda.max_memory_allocated() / _MIB, float(loss)


def _padding_stats(batch: Batch) -> Dict[str, float]:
    """Describe how much of the padded rectangle is real.

    Args:
        batch: an assembled batch.

    Returns:
        Dict[str, float]: row count, max and mean length, and padding waste.
    """
    cu_seqlens = batch.additional_infos["cu_seqlens"]
    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).cpu().numpy()
    padded = int(lengths.max()) * len(lengths)
    return {
        "rows": int(len(lengths)),
        "max_len": int(lengths.max()),
        "mean_len": float(lengths.mean()),
        "real_tokens": int(lengths.sum()),
        "padded_tokens": int(padded),
        "pad_waste_pct": float(100.0 * (1.0 - lengths.sum() / padded)),
    }


def _run(args: argparse.Namespace) -> List[Dict[str, object]]:
    """Benchmark every requested kernel at every requested batch size.

    Args:
        args: parsed command line.

    Returns:
        List[Dict[str, object]]: one record per (batch size, kernel).
    """
    device = torch.device(args.device)
    kernels = [k.strip() for k in args.kernels.split(",") if k.strip()]
    records: List[Dict[str, object]] = []
    for batch_size in [int(x) for x in args.batch_sizes.split(",")]:
        rows = _read_rows(args, batch_size, args.seed)
        stats: Optional[Dict[str, float]] = None
        baseline: Optional[float] = None
        for name in kernels:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            model, compiled_prompt = _build_model(
                args, GenRecModelConfig.AttnKernel.Value(name), device
            )
            batch = _build_batch(args, compiled_prompt, rows, device)
            stats = stats or _padding_stats(batch)
            if stats is not None and baseline is None:
                print(
                    "\n== batch={rows} max_len={max_len} mean_len={mean_len:.1f} "
                    "pad_waste={pad_waste_pct:.1f}% ==".format(**stats),
                    flush=True,
                )
            step_ms, peak_mib, loss = _time_steps(
                model, batch, args.iters, args.warmup, args.autocast
            )
            baseline = baseline or step_ms
            records.append(
                dict(
                    kernel=name,
                    step_ms=step_ms,
                    peak_mib=peak_mib,
                    loss=loss,
                    speedup=baseline / step_ms,
                    **stats,
                )
            )
            print(
                f"  {name:18s} step {step_ms:9.2f} ms  peak {peak_mib:9.1f} MiB  "
                f"loss {loss:8.4f}  {baseline / step_ms:5.2f}x",
                flush=True,
            )
            del model, batch
            torch.cuda.empty_cache()
    return records


def main() -> None:
    """Parse arguments and report the per-kernel training step cost."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True, help="HF directory of the LM")
    parser.add_argument("--tokenizer", default=None, help="tokenizer.json path")
    parser.add_argument("--data-glob", required=True, help="sample parquet glob")
    parser.add_argument("--hist-feature", default="item_list_with_sid")
    parser.add_argument("--label-field", default="single_label")
    parser.add_argument("--codebook", default="256,256,256")
    parser.add_argument("--vocab-pad-to-multiple-of", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=900)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--beam-widths", default="50,50,50")
    parser.add_argument("--num-return-sequences", type=int, default=50)
    parser.add_argument("--dtype", default="FP32", choices=["FP32", "BF16", "FP16"])
    parser.add_argument(
        "--autocast",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="bfloat16 autocast, matching train_config.mixed_precision BF16",
    )
    parser.add_argument("--kernels", default="SDPA,FLASH_ATTENTION_2")
    parser.add_argument("--batch-sizes", default="20")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--json-output", default=None)
    parser.add_argument(
        "--prompt",
        default=(
            "system\n你是一个推荐系统，根据用户的历史行为，预测用户在长视频场景的下一步行为。"
            "我会给你一串连续行为的语义编码，按照用户观看的时间顺序排列，每个行为用三个词表示。\n"
            "user\n用户历史行为的语义编码如下：{{HIST}}。"
            "请预测下一步行为的三个语义编码。\nassistant:\n"
        ),
        help="prompt template; {{HIST}} is replaced by the history slot",
    )
    args = parser.parse_args()

    print(f"device={torch.cuda.get_device_name(0)} torch={torch.__version__}")
    print(f"backbone={args.backbone} dtype={args.dtype} autocast={args.autocast}")
    records = _run(args)
    if args.json_output:
        with open(args.json_output, "w") as output_file:
            json.dump(
                dict(
                    gpu=torch.cuda.get_device_name(0),
                    torch=torch.__version__,
                    backbone=args.backbone,
                    dtype=args.dtype,
                    autocast=args.autocast,
                    records=records,
                ),
                output_file,
                indent=2,
            )


if __name__ == "__main__":
    main()
