# Copyright (c) 2026, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Offline SID codebook collision-prevention tool.

The v1 allocator is a deterministic post-process over predicted SID rows plus
explicit candidate SID rows. It does not generate random fallback candidates,
and candidate output from SID models is expected to be opt-in upstream.
"""

import argparse
import hashlib
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pyarrow as pa

# Register the local reader/writer classes used through create_reader/writer.
import tzrec.datasets.csv_dataset  # noqa: F401
import tzrec.datasets.parquet_dataset  # noqa: F401
from tzrec.datasets.dataset import create_reader, create_writer
from tzrec.utils.logging_util import logger


@dataclass(frozen=True)
class RawSidRow:
    """One raw item -> SID row."""

    item_id: Any
    item_key: str
    origin_codebook: str


@dataclass(frozen=True)
class CandidateSidRow:
    """One item -> candidate SID row."""

    item_key: str
    candidate_codebook: str
    priority: int
    score: float


@dataclass(frozen=True)
class AssignedSidRow:
    """One final item -> SID assignment row."""

    item_id: Any
    item_key: str
    origin_codebook: str
    codebook: str
    index: int


@dataclass(frozen=True)
class AssignmentStats:
    """Summary statistics for a collision-prevention run."""

    total_items: int
    raw_collision_buckets: int
    final_collision_buckets: int
    reassigned_count: int
    unassigned_count: int
    iteration_count: int
    max_final_bucket_size: int


@dataclass(frozen=True)
class OdpsTableRef:
    """Parsed ODPS table URI."""

    project: str
    table: str
    partitions: Tuple[str, ...]
    schema: Optional[str] = None


def _stable_hash(*parts: Any) -> int:
    h = hashlib.blake2b(digest_size=8)
    for part in parts:
        h.update(str(part).encode("utf-8"))
        h.update(b"\x1f")
    return int.from_bytes(h.digest(), byteorder="big", signed=False)


def _cell_to_code(value: Any, delimiter: str) -> str:
    if value is None:
        raise ValueError("SID code value cannot be null.")
    if isinstance(value, (list, tuple)):
        return delimiter.join(str(v) for v in value)
    return str(value)


def _split_compact_candidates(value: Any, delimiter: str) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [_cell_to_code(v, ",") for v in value if v is not None]
    text = str(value).strip()
    if not text:
        return []
    return [part.strip() for part in text.split(delimiter) if part.strip()]


def _array_to_pylist(batch: Dict[str, pa.Array], field: str) -> List[Any]:
    if field not in batch:
        raise ValueError(
            f"required field {field!r} not found; available fields: "
            f"{sorted(batch.keys())}"
        )
    return batch[field].to_pylist()


def _read_batches(
    input_path: str,
    reader_type: Optional[str],
    batch_size: int,
    quota_name: str,
) -> Iterable[Dict[str, pa.Array]]:
    reader = create_reader(
        input_path=input_path,
        batch_size=batch_size,
        reader_type=reader_type,
        quota_name=quota_name,
    )
    yield from reader.to_batches()


def _load_raw_sid_rows(
    input_path: str,
    item_id_field: str,
    code_field: Optional[str],
    code_fields: Optional[Sequence[str]],
    code_delimiter: str,
    reader_type: Optional[str],
    batch_size: int,
    quota_name: str,
) -> List[RawSidRow]:
    rows: List[RawSidRow] = []
    seen = set()
    if bool(code_field) == bool(code_fields):
        raise ValueError("Set exactly one of --code_field or --code_fields.")

    for batch in _read_batches(input_path, reader_type, batch_size, quota_name):
        item_ids = _array_to_pylist(batch, item_id_field)
        if code_field:
            codes = _array_to_pylist(batch, code_field)
            origin_codes = [_cell_to_code(v, code_delimiter) for v in codes]
        else:
            assert code_fields is not None
            code_columns = [_array_to_pylist(batch, f) for f in code_fields]
            origin_codes = [
                code_delimiter.join(str(col[i]) for col in code_columns)
                for i in range(len(item_ids))
            ]

        for item_id, origin_codebook in zip(item_ids, origin_codes):
            item_key = str(item_id)
            if item_key in seen:
                raise ValueError(f"duplicate item_id in raw SID input: {item_key}")
            seen.add(item_key)
            rows.append(
                RawSidRow(
                    item_id=item_id,
                    item_key=item_key,
                    origin_codebook=origin_codebook,
                )
            )

    if not rows:
        raise ValueError("raw SID input is empty.")
    return rows


def _load_candidate_rows(
    input_path: str,
    item_id_field: str,
    candidate_codebook_field: Optional[str],
    compact_candidate_field: Optional[str],
    priority_field: Optional[str],
    score_field: Optional[str],
    code_delimiter: str,
    candidate_delimiter: str,
    reader_type: Optional[str],
    batch_size: int,
    quota_name: str,
) -> List[CandidateSidRow]:
    rows: List[CandidateSidRow] = []
    if bool(candidate_codebook_field) == bool(compact_candidate_field):
        raise ValueError(
            "Set exactly one of --candidate_codebook_field or "
            "--compact_candidate_field."
        )

    for batch in _read_batches(input_path, reader_type, batch_size, quota_name):
        item_ids = _array_to_pylist(batch, item_id_field)
        priorities = (
            _array_to_pylist(batch, priority_field)
            if priority_field and priority_field in batch
            else None
        )
        scores = (
            _array_to_pylist(batch, score_field)
            if score_field and score_field in batch
            else None
        )
        if candidate_codebook_field:
            candidates = _array_to_pylist(batch, candidate_codebook_field)
            for i, (item_id, candidate) in enumerate(zip(item_ids, candidates)):
                rows.append(
                    CandidateSidRow(
                        item_key=str(item_id),
                        candidate_codebook=_cell_to_code(candidate, code_delimiter),
                        priority=int(priorities[i]) if priorities is not None else 1,
                        score=float(scores[i]) if scores is not None else 0.0,
                    )
                )
        else:
            assert compact_candidate_field is not None
            compact_values = _array_to_pylist(batch, compact_candidate_field)
            for item_id, compact_value in zip(item_ids, compact_values):
                for priority, candidate in enumerate(
                    _split_compact_candidates(compact_value, candidate_delimiter),
                    start=1,
                ):
                    rows.append(
                        CandidateSidRow(
                            item_key=str(item_id),
                            candidate_codebook=candidate,
                            priority=priority,
                            score=0.0,
                        )
                    )

    return rows


def _raw_collision_buckets(raw_rows: Sequence[RawSidRow], capacity: int) -> int:
    counts: Dict[str, int] = defaultdict(int)
    for row in raw_rows:
        counts[row.origin_codebook] += 1
    return sum(1 for count in counts.values() if count > capacity)


def _assignment_sort_key(seed: int, row: RawSidRow) -> Tuple[int, str]:
    return (_stable_hash(seed, row.origin_codebook, row.item_key), row.item_key)


def _candidate_sort_key(
    seed: int,
    row: CandidateSidRow,
    score_order: str,
) -> Tuple[int, float, int, str, str]:
    score = row.score if score_order == "lower" else -row.score
    return (
        row.priority,
        score,
        _stable_hash(seed, row.item_key, row.candidate_codebook),
        row.item_key,
        row.candidate_codebook,
    )


def assign_sid_collisions(
    raw_rows: Sequence[RawSidRow],
    candidate_rows: Sequence[CandidateSidRow],
    capacity: int,
    max_iters: int = 50,
    seed: int = 2026,
    score_order: str = "lower",
    unassigned_policy: str = "error",
) -> Tuple[List[AssignedSidRow], AssignmentStats]:
    """Assign overflow SID rows to non-full candidate codebooks.

    Args:
        raw_rows: item -> raw SID rows.
        candidate_rows: explicit candidate rows. No random fallback is generated.
        capacity: max items per final codebook.
        max_iters: max greedy assignment rounds.
        seed: deterministic tie-breaker seed.
        score_order: "lower" for distances, "higher" for similarities.
        unassigned_policy: "error", "drop", or "keep_original".

    Returns:
        Final assignments and summary stats.
    """
    if capacity < 1:
        raise ValueError(f"capacity must be >= 1, got {capacity}")
    if score_order not in ("lower", "higher"):
        raise ValueError("score_order must be 'lower' or 'higher'.")
    if unassigned_policy not in ("error", "drop", "keep_original"):
        raise ValueError(
            "unassigned_policy must be one of: error, drop, keep_original."
        )
    if len({row.item_key for row in raw_rows}) != len(raw_rows):
        raise ValueError("raw_rows contains duplicate item_id values.")

    raw_by_item = {row.item_key: row for row in raw_rows}
    by_origin: Dict[str, List[RawSidRow]] = defaultdict(list)
    for row in raw_rows:
        by_origin[row.origin_codebook].append(row)

    assigned: List[AssignedSidRow] = []
    assigned_items = set()
    code_counts: Dict[str, int] = defaultdict(int)

    for codebook, rows in by_origin.items():
        for index, row in enumerate(
            sorted(rows, key=lambda r: _assignment_sort_key(seed, r))[:capacity],
            start=1,
        ):
            assigned.append(
                AssignedSidRow(
                    item_id=row.item_id,
                    item_key=row.item_key,
                    origin_codebook=row.origin_codebook,
                    codebook=codebook,
                    index=index,
                )
            )
            assigned_items.add(row.item_key)
            code_counts[codebook] += 1

    dedup_candidates: Dict[Tuple[str, str], CandidateSidRow] = {}
    for row in candidate_rows:
        if row.item_key in raw_by_item:
            key = (row.item_key, row.candidate_codebook)
            current = dedup_candidates.get(key)
            if current is None or _candidate_sort_key(
                seed, row, score_order
            ) < _candidate_sort_key(seed, current, score_order):
                dedup_candidates[key] = row

    candidates_by_item: Dict[str, List[CandidateSidRow]] = defaultdict(list)
    for row in dedup_candidates.values():
        candidates_by_item[row.item_key].append(row)

    overflow_items = set(raw_by_item) - assigned_items
    if overflow_items and not candidate_rows:
        raise ValueError(
            "raw SID input has overflow rows, but no explicit candidate input was "
            "provided."
        )

    iteration_count = 0
    for iteration in range(max_iters):
        unassigned = set(raw_by_item) - assigned_items
        if not unassigned:
            break

        available: List[CandidateSidRow] = []
        for item_key in unassigned:
            for candidate in candidates_by_item.get(item_key, []):
                if code_counts[candidate.candidate_codebook] < capacity:
                    available.append(candidate)
        if not available:
            break

        selected_by_codebook: List[CandidateSidRow] = []
        by_codebook: Dict[str, List[CandidateSidRow]] = defaultdict(list)
        for candidate in available:
            by_codebook[candidate.candidate_codebook].append(candidate)

        for codebook, rows in by_codebook.items():
            remaining = capacity - code_counts[codebook]
            if remaining <= 0:
                continue
            selected_by_codebook.extend(
                sorted(
                    rows,
                    key=lambda r: _candidate_sort_key(seed, r, score_order),
                )[:remaining]
            )

        best_by_item: Dict[str, CandidateSidRow] = {}
        for candidate in selected_by_codebook:
            current = best_by_item.get(candidate.item_key)
            if current is None or _candidate_sort_key(
                seed, candidate, score_order
            ) < _candidate_sort_key(seed, current, score_order):
                best_by_item[candidate.item_key] = candidate

        accepted = sorted(
            best_by_item.values(),
            key=lambda r: (
                r.candidate_codebook,
                _candidate_sort_key(seed, r, score_order),
            ),
        )
        progress = 0
        for candidate in accepted:
            if candidate.item_key in assigned_items:
                continue
            if code_counts[candidate.candidate_codebook] >= capacity:
                continue
            raw = raw_by_item[candidate.item_key]
            code_counts[candidate.candidate_codebook] += 1
            assigned.append(
                AssignedSidRow(
                    item_id=raw.item_id,
                    item_key=raw.item_key,
                    origin_codebook=raw.origin_codebook,
                    codebook=candidate.candidate_codebook,
                    index=code_counts[candidate.candidate_codebook],
                )
            )
            assigned_items.add(candidate.item_key)
            progress += 1

        iteration_count = iteration + 1
        if progress == 0:
            break

    unassigned = sorted(set(raw_by_item) - assigned_items)
    if unassigned:
        if unassigned_policy == "error":
            preview = ",".join(unassigned[:10])
            raise RuntimeError(
                f"{len(unassigned)} items could not be assigned within capacity; "
                f"first unassigned item_ids: {preview}"
            )
        if unassigned_policy == "keep_original":
            for item_key in unassigned:
                raw = raw_by_item[item_key]
                code_counts[raw.origin_codebook] += 1
                assigned.append(
                    AssignedSidRow(
                        item_id=raw.item_id,
                        item_key=raw.item_key,
                        origin_codebook=raw.origin_codebook,
                        codebook=raw.origin_codebook,
                        index=code_counts[raw.origin_codebook],
                    )
                )

    final_counts: Dict[str, int] = defaultdict(int)
    for row in assigned:
        final_counts[row.codebook] += 1
    stats = AssignmentStats(
        total_items=len(raw_rows),
        raw_collision_buckets=_raw_collision_buckets(raw_rows, capacity),
        final_collision_buckets=sum(
            1 for count in final_counts.values() if count > capacity
        ),
        reassigned_count=sum(
            1 for row in assigned if row.origin_codebook != row.codebook
        ),
        unassigned_count=len(unassigned) if unassigned_policy != "keep_original" else 0,
        iteration_count=iteration_count,
        max_final_bucket_size=max(final_counts.values()) if final_counts else 0,
    )
    assigned = sorted(assigned, key=lambda r: (r.codebook, r.index, r.item_key))
    return assigned, stats


def _item_id_array(rows: Sequence[AssignedSidRow]) -> pa.Array:
    values = [row.item_id for row in rows]
    if all(isinstance(v, int) and not isinstance(v, bool) for v in values):
        return pa.array(values, type=pa.int64())
    return pa.array([str(v) for v in values], type=pa.string())


def _write_assignments(
    rows: Sequence[AssignedSidRow],
    output_path: str,
    writer_type: str,
    quota_name: str,
) -> None:
    writer = create_writer(
        output_path,
        writer_type=writer_type,
        quota_name=quota_name,
        world_size=1,
    )
    writer.write(
        OrderedDict(
            [
                ("item_id", _item_id_array(rows)),
                (
                    "origin_codebook",
                    pa.array([row.origin_codebook for row in rows], type=pa.string()),
                ),
                (
                    "codebook",
                    pa.array([row.codebook for row in rows], type=pa.string()),
                ),
                ("index", pa.array([row.index for row in rows], type=pa.int64())),
            ]
        )
    )
    writer.close()


def _write_diagnostics(
    stats: AssignmentStats,
    output_path: str,
    writer_type: str,
    quota_name: str,
) -> None:
    writer = create_writer(
        output_path,
        writer_type=writer_type,
        quota_name=quota_name,
        world_size=1,
    )
    writer.write(
        OrderedDict(
            [
                ("total_items", pa.array([stats.total_items], type=pa.int64())),
                (
                    "raw_collision_buckets",
                    pa.array([stats.raw_collision_buckets], type=pa.int64()),
                ),
                (
                    "final_collision_buckets",
                    pa.array([stats.final_collision_buckets], type=pa.int64()),
                ),
                (
                    "reassigned_count",
                    pa.array([stats.reassigned_count], type=pa.int64()),
                ),
                (
                    "unassigned_count",
                    pa.array([stats.unassigned_count], type=pa.int64()),
                ),
                (
                    "iteration_count",
                    pa.array([stats.iteration_count], type=pa.int64()),
                ),
                (
                    "max_final_bucket_size",
                    pa.array([stats.max_final_bucket_size], type=pa.int64()),
                ),
            ]
        )
    )
    writer.close()


def run_local(args: argparse.Namespace) -> AssignmentStats:
    """Run local CSV/Parquet collision prevention."""
    code_fields = (
        [field.strip() for field in args.code_fields.split(",") if field.strip()]
        if args.code_fields
        else None
    )
    code_field = None if code_fields else args.code_field
    raw_rows = _load_raw_sid_rows(
        input_path=args.input_path,
        item_id_field=args.item_id_field,
        code_field=code_field,
        code_fields=code_fields,
        code_delimiter=args.code_delimiter,
        reader_type=args.reader_type,
        batch_size=args.batch_size,
        quota_name=args.odps_data_quota_name,
    )
    candidate_rows: List[CandidateSidRow] = []
    if args.candidate_input_path:
        candidate_codebook_field = (
            None if args.compact_candidate_field else args.candidate_codebook_field
        )
        candidate_rows = _load_candidate_rows(
            input_path=args.candidate_input_path,
            item_id_field=args.candidate_item_id_field or args.item_id_field,
            candidate_codebook_field=candidate_codebook_field,
            compact_candidate_field=args.compact_candidate_field,
            priority_field=args.priority_field,
            score_field=args.score_field,
            code_delimiter=args.code_delimiter,
            candidate_delimiter=args.candidate_delimiter,
            reader_type=args.candidate_reader_type or args.reader_type,
            batch_size=args.batch_size,
            quota_name=args.odps_data_quota_name,
        )

    rows, stats = assign_sid_collisions(
        raw_rows=raw_rows,
        candidate_rows=candidate_rows,
        capacity=args.max_items_per_codebook,
        max_iters=args.max_iters,
        seed=args.seed,
        score_order=args.score_order,
        unassigned_policy=args.unassigned_policy,
    )
    _write_assignments(
        rows,
        args.output_path,
        args.writer_type,
        args.odps_data_quota_name,
    )
    if args.diagnostics_output_path:
        _write_diagnostics(
            stats,
            args.diagnostics_output_path,
            args.writer_type,
            args.odps_data_quota_name,
        )
    logger.info("SID collision prevention finished: %s", stats)
    return stats


def _parse_odps_table(path: str) -> OdpsTableRef:
    parts = path.split("/")
    if len(parts) < 5 or parts[0] != "odps:" or parts[3] != "tables":
        raise ValueError(
            f"invalid ODPS path {path!r}; expected "
            "odps://project/tables/table[/pt=value]"
        )
    table = parts[4]
    schema = None
    if "." in table:
        schema, table = table.split(".", 1)
    return OdpsTableRef(
        project=parts[2],
        table=table,
        partitions=tuple(p for p in parts[5:] if p),
        schema=schema,
    )


def _odps_table_name(ref: OdpsTableRef) -> str:
    table = f"{ref.schema}.{ref.table}" if ref.schema else ref.table
    return f"{ref.project}.{table}"


def _odps_partition_predicate(ref: OdpsTableRef) -> str:
    if not ref.partitions:
        return ""
    predicates = []
    for part in ref.partitions:
        if "=" not in part:
            raise ValueError(f"invalid ODPS partition segment: {part!r}")
        key, value = part.split("=", 1)
        predicates.append(f"{key}='{value}'")
    return " AND " + " AND ".join(predicates)


def _odps_insert_target(ref: OdpsTableRef) -> str:
    if not ref.partitions:
        return _odps_table_name(ref)
    specs = []
    for part in ref.partitions:
        key, value = part.split("=", 1)
        specs.append(f"{key}='{value}'")
    return f"{_odps_table_name(ref)} PARTITION ({','.join(specs)})"


def _odps_partition_schema(ref: OdpsTableRef) -> str:
    if not ref.partitions:
        return ""
    fields = []
    for part in ref.partitions:
        key, _ = part.split("=", 1)
        fields.append(f"{key} STRING")
    return f" PARTITIONED BY ({','.join(fields)})"


def generate_odps_sql(args: argparse.Namespace) -> List[str]:
    """Generate deterministic MaxCompute SQL for canonical candidate tables."""
    if not args.candidate_input_path:
        raise ValueError("--candidate_input_path is required for --backend odps.")
    if args.code_fields:
        raise ValueError(
            "--backend odps currently supports --code_field, not split code_fields."
        )
    if args.compact_candidate_field:
        raise ValueError(
            "--backend odps expects canonical candidate rows; compact candidates "
            "are supported in local CSV/Parquet mode."
        )

    raw_ref = _parse_odps_table(args.input_path)
    candidate_ref = _parse_odps_table(args.candidate_input_path)
    output_ref = _parse_odps_table(args.output_path)
    prefix = args.temp_prefix or "tmp_sid_collision"
    lifecycle = args.odps_lifecycle
    assigned = f"{prefix}_assigned"
    selected = f"{prefix}_selected"
    counts = f"{prefix}_counts"
    raw_table = _odps_table_name(raw_ref)
    cand_table = _odps_table_name(candidate_ref)
    output_target = _odps_insert_target(output_ref)
    output_table = _odps_table_name(output_ref)
    raw_predicate = _odps_partition_predicate(raw_ref)
    cand_predicate = _odps_partition_predicate(candidate_ref)
    score_expr = f"CAST({args.score_field} AS DOUBLE)" if args.score_field else "0.0"
    priority_expr = (
        f"CAST({args.priority_field} AS BIGINT)" if args.priority_field else "1"
    )
    score_order = "score ASC" if args.score_order == "lower" else "score DESC"

    sqls = [
        "SET odps.sql.type.system.odps2=true",
        (
            f"CREATE TABLE IF NOT EXISTS {assigned} ("
            "item_id STRING, origin_codebook STRING, codebook STRING, `index` BIGINT"
            f") LIFECYCLE {lifecycle}"
        ),
        (
            f"CREATE TABLE IF NOT EXISTS {selected} ("
            "item_id STRING, origin_codebook STRING, codebook STRING, "
            "priority BIGINT, score DOUBLE"
            f") LIFECYCLE {lifecycle}"
        ),
        (
            f"CREATE TABLE IF NOT EXISTS {counts} ("
            "codebook STRING, cnt BIGINT"
            f") LIFECYCLE {lifecycle}"
        ),
        (
            f"CREATE TABLE IF NOT EXISTS {output_table} ("
            "item_id STRING, origin_codebook STRING, codebook STRING, `index` BIGINT"
            f"){_odps_partition_schema(output_ref)}"
        ),
        (
            f"INSERT OVERWRITE TABLE {assigned}\n"
            "SELECT item_id, origin_codebook, origin_codebook AS codebook,\n"
            "       rn AS `index`\n"
            "FROM (\n"
            f"  SELECT CAST({args.item_id_field} AS STRING) AS item_id,\n"
            f"         CAST({args.code_field} AS STRING) AS origin_codebook,\n"
            "         ROW_NUMBER() OVER (\n"
            f"           PARTITION BY CAST({args.code_field} AS STRING)\n"
            "           ORDER BY ABS(HASH(CONCAT("
            f"'{args.seed}', ':', CAST({args.item_id_field} AS STRING))))\n"
            "         ) AS rn\n"
            f"  FROM {raw_table}\n"
            f"  WHERE 1=1{raw_predicate}\n"
            ") t\n"
            f"WHERE rn <= {args.max_items_per_codebook}"
        ),
    ]
    for i in range(args.max_iters):
        sqls.extend(
            [
                (
                    f"INSERT OVERWRITE TABLE {counts}\n"
                    "SELECT codebook, COUNT(*) AS cnt\n"
                    f"FROM {assigned}\n"
                    "GROUP BY codebook"
                ),
                (
                    f"INSERT OVERWRITE TABLE {selected}\n"
                    "SELECT item_id, origin_codebook, codebook, priority, score\n"
                    "FROM (\n"
                    "  SELECT c.item_id, c.origin_codebook, c.codebook, "
                    "c.priority, c.score,\n"
                    "         ROW_NUMBER() OVER (\n"
                    "           PARTITION BY c.codebook\n"
                    f"           ORDER BY c.priority ASC, c.{score_order}, "
                    "ABS(HASH(CONCAT("
                    f"'{args.seed}', ':', c.item_id, ':', c.codebook)))\n"
                    "         ) AS rn,\n"
                    "         COALESCE(cnt.cnt, 0) AS current_cnt\n"
                    "  FROM (\n"
                    "    SELECT CAST("
                    f"{args.candidate_item_id_field or args.item_id_field}"
                    " AS STRING) AS item_id,\n"
                    "           CAST("
                    f"{args.candidate_origin_codebook_field} AS STRING"
                    ") AS origin_codebook,\n"
                    "           CAST("
                    f"{args.candidate_codebook_field}"
                    " AS STRING) AS codebook,\n"
                    f"           {priority_expr} AS priority,\n"
                    f"           {score_expr} AS score\n"
                    f"    FROM {cand_table}\n"
                    f"    WHERE 1=1{cand_predicate}\n"
                    "  ) c\n"
                    "  INNER JOIN (\n"
                    f"    SELECT CAST({args.item_id_field} AS STRING) AS item_id\n"
                    f"    FROM {raw_table}\n"
                    f"    WHERE 1=1{raw_predicate}\n"
                    "  ) r ON c.item_id = r.item_id\n"
                    f"  LEFT OUTER JOIN {assigned} a ON c.item_id = a.item_id\n"
                    f"  LEFT OUTER JOIN {counts} cnt ON c.codebook = cnt.codebook\n"
                    "  WHERE a.item_id IS NULL\n"
                    f"    AND COALESCE(cnt.cnt, 0) < {args.max_items_per_codebook}\n"
                    ") ranked\n"
                    f"WHERE rn <= {args.max_items_per_codebook} - current_cnt"
                ),
                (
                    f"INSERT INTO TABLE {assigned}\n"
                    "SELECT item_id, origin_codebook, codebook,\n"
                    "       current_cnt + codebook_rn AS `index`\n"
                    "FROM (\n"
                    "  SELECT s.item_id, s.origin_codebook, s.codebook, "
                    "s.priority, s.score,\n"
                    "         ROW_NUMBER() OVER (\n"
                    "           PARTITION BY s.item_id\n"
                    f"           ORDER BY s.priority ASC, s.{score_order}, "
                    "ABS(HASH(CONCAT("
                    f"'{args.seed}', ':', s.item_id, ':', s.codebook)))\n"
                    "         ) AS item_rn,\n"
                    "         ROW_NUMBER() OVER (\n"
                    "           PARTITION BY s.codebook\n"
                    f"           ORDER BY s.priority ASC, s.{score_order}, "
                    "ABS(HASH(CONCAT("
                    f"'{args.seed}', ':', s.item_id, ':', s.codebook)))\n"
                    "         ) AS codebook_rn,\n"
                    "         COALESCE(cnt.cnt, 0) AS current_cnt\n"
                    f"  FROM {selected} s\n"
                    f"  LEFT OUTER JOIN {counts} cnt ON s.codebook = cnt.codebook\n"
                    ") x\n"
                    "WHERE item_rn = 1\n"
                    f"  AND current_cnt + codebook_rn <= {args.max_items_per_codebook}"
                ),
                (
                    f"-- iteration {i + 1}: stop early if this returns 0\n"
                    "SELECT COUNT(*) AS remaining_unassigned\n"
                    f"FROM {raw_table} r\n"
                    f"LEFT OUTER JOIN {assigned} a\n"
                    f"ON CAST(r.{args.item_id_field} AS STRING) = a.item_id\n"
                    f"WHERE a.item_id IS NULL{raw_predicate}"
                ),
            ]
        )
    sqls.append(
        f"INSERT OVERWRITE TABLE {output_target}\n"
        "SELECT item_id, origin_codebook, codebook, `index`\n"
        f"FROM {assigned}"
    )
    return sqls


def run_odps(args: argparse.Namespace) -> None:
    """Run MaxCompute SQL collision prevention."""
    sqls = generate_odps_sql(args)
    if args.dry_run_sql:
        print(";\n\n".join(sqls) + ";")
        return

    from odps import ODPS

    from tzrec.datasets.odps_dataset import _create_odps_account

    output_ref = _parse_odps_table(args.output_path)
    account, endpoint = _create_odps_account()
    odps = ODPS(account=account, project=output_ref.project, endpoint=endpoint)
    for sql in sqls:
        logger.info("Executing ODPS SQL:\n%s", sql)
        odps.execute_sql(sql).wait_for_success()


def build_parser() -> argparse.ArgumentParser:
    """Build the command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Prevent SID codebook collisions with explicit candidates."
    )
    parser.add_argument("--backend", choices=["local", "odps"], default="local")
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--candidate_input_path", default=None)
    parser.add_argument("--diagnostics_output_path", default=None)
    parser.add_argument(
        "--reader_type", choices=["CsvReader", "ParquetReader"], default=None
    )
    parser.add_argument(
        "--candidate_reader_type",
        choices=["CsvReader", "ParquetReader"],
        default=None,
    )
    parser.add_argument(
        "--writer_type",
        choices=["CsvWriter", "ParquetWriter"],
        default="ParquetWriter",
    )
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--item_id_field", default="item_id")
    parser.add_argument("--candidate_item_id_field", default=None)
    parser.add_argument("--code_field", default="codes")
    parser.add_argument(
        "--code_fields",
        default=None,
        help="Comma-separated split code fields. Mutually exclusive with code_field.",
    )
    parser.add_argument("--code_delimiter", default=",")
    parser.add_argument("--candidate_codebook_field", default="candidate_codebook")
    parser.add_argument("--candidate_origin_codebook_field", default="origin_codebook")
    parser.add_argument(
        "--compact_candidate_field",
        default=None,
        help="Compact string/list candidate field for local mode.",
    )
    parser.add_argument(
        "--candidate_delimiter",
        default="|",
        help="Delimiter for compact string candidate lists.",
    )
    parser.add_argument("--priority_field", default="priority")
    parser.add_argument("--score_field", default="score")
    parser.add_argument("--score_order", choices=["lower", "higher"], default="lower")
    parser.add_argument("--max_items_per_codebook", type=int, required=True)
    parser.add_argument("--max_iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--unassigned_policy",
        choices=["error", "drop", "keep_original"],
        default="error",
    )
    parser.add_argument("--odps_data_quota_name", default="pay-as-you-go")
    parser.add_argument("--temp_prefix", default=None)
    parser.add_argument("--odps_lifecycle", type=int, default=7)
    parser.add_argument("--dry_run_sql", action="store_true", default=False)
    return parser


def main() -> None:
    """Command line entrypoint."""
    parser = build_parser()
    args = parser.parse_args()
    if args.backend == "local":
        run_local(args)
    else:
        run_odps(args)


if __name__ == "__main__":
    main()
