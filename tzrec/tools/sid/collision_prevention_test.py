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

import os
import shutil
import sys
import tempfile
import types
import unittest
from collections import Counter
from unittest import mock

import pyarrow as pa
from pyarrow import csv, parquet

from tzrec.tools.sid.collision_prevention import (
    CandidateSidRow,
    RawSidRow,
    assign_sid_collisions,
    build_parser,
    generate_odps_sql,
    run_local,
    run_odps,
)


class SidCollisionPreventionTest(unittest.TestCase):
    def setUp(self) -> None:
        if not os.path.exists("./tmp"):
            os.makedirs("./tmp")
        self.test_dir = tempfile.mkdtemp(prefix="tzrec_", dir="./tmp")

    def tearDown(self) -> None:
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_assign_sid_collisions_respects_capacity(self) -> None:
        raw_rows = [
            RawSidRow("item_0", "item_0", "A"),
            RawSidRow("item_1", "item_1", "A"),
            RawSidRow("item_2", "item_2", "A"),
            RawSidRow("item_3", "item_3", "B"),
        ]
        candidate_rows = [
            CandidateSidRow("item_0", "C", 1, 0.1),
            CandidateSidRow("item_1", "C", 1, 0.1),
            CandidateSidRow("item_2", "C", 1, 0.1),
        ]

        assigned, stats = assign_sid_collisions(
            raw_rows,
            candidate_rows,
            capacity=2,
            seed=7,
        )

        self.assertEqual(len(assigned), 4)
        self.assertEqual(len({row.item_key for row in assigned}), 4)
        self.assertLessEqual(max(Counter(row.codebook for row in assigned).values()), 2)
        self.assertEqual(stats.raw_collision_buckets, 1)
        self.assertEqual(stats.final_collision_buckets, 0)
        self.assertEqual(stats.reassigned_count, 1)
        self.assertEqual(stats.unassigned_count, 0)

    def test_missing_candidates_errors_on_overflow(self) -> None:
        raw_rows = [
            RawSidRow("item_0", "item_0", "A"),
            RawSidRow("item_1", "item_1", "A"),
        ]
        with self.assertRaisesRegex(ValueError, "no explicit candidate input"):
            assign_sid_collisions(raw_rows, [], capacity=1)

    def test_duplicate_candidates_do_not_consume_capacity_twice(self) -> None:
        raw_rows = [
            RawSidRow("item_0", "item_0", "A"),
            RawSidRow("item_1", "item_1", "A"),
            RawSidRow("item_2", "item_2", "A"),
        ]
        candidate_rows = [
            CandidateSidRow("item_0", "C", 1, 0.1),
            CandidateSidRow("item_0", "C", 2, 0.2),
            CandidateSidRow("item_1", "C", 1, 0.1),
            CandidateSidRow("item_2", "C", 1, 0.1),
        ]

        assigned, stats = assign_sid_collisions(
            raw_rows,
            candidate_rows,
            capacity=2,
            seed=7,
        )

        self.assertEqual(len(assigned), 3)
        self.assertEqual(stats.unassigned_count, 0)
        self.assertLessEqual(max(Counter(row.codebook for row in assigned).values()), 2)

    def test_local_csv_outputs_codebooks_as_strings(self) -> None:
        raw_path = os.path.join(self.test_dir, "raw.csv")
        cand_path = os.path.join(self.test_dir, "cand.csv")
        out_dir = os.path.join(self.test_dir, "out")
        csv.write_csv(
            pa.table(
                {
                    "item_id": ["1", "2", "3"],
                    "codes": ["A", "A", "A"],
                }
            ),
            raw_path,
        )
        csv.write_csv(
            pa.table(
                {
                    "item_id": ["1", "2", "3"],
                    "candidate_codebook": ["C", "C", "C"],
                    "priority": [1, 1, 1],
                    "score": [0.1, 0.1, 0.1],
                }
            ),
            cand_path,
        )

        args = build_parser().parse_args(
            [
                "--input_path",
                raw_path,
                "--candidate_input_path",
                cand_path,
                "--output_path",
                out_dir,
                "--reader_type",
                "CsvReader",
                "--writer_type",
                "CsvWriter",
                "--max_items_per_codebook",
                "2",
            ]
        )
        stats = run_local(args)

        self.assertEqual(stats.reassigned_count, 1)
        result = csv.read_csv(os.path.join(out_dir, "part-0.csv"))
        self.assertEqual(result.schema.field("origin_codebook").type, pa.string())
        self.assertEqual(result.schema.field("codebook").type, pa.string())
        self.assertLessEqual(max(Counter(result["codebook"].to_pylist()).values()), 2)

    def test_local_parquet_accepts_list_codes(self) -> None:
        raw_path = os.path.join(self.test_dir, "raw.parquet")
        cand_path = os.path.join(self.test_dir, "cand.parquet")
        out_dir = os.path.join(self.test_dir, "out_parquet")
        parquet.write_table(
            pa.table(
                {
                    "item_id": pa.array([1, 2, 3], type=pa.int64()),
                    "codes": pa.array([[1, 2], [1, 2], [1, 2]]),
                }
            ),
            raw_path,
        )
        parquet.write_table(
            pa.table(
                {
                    "item_id": pa.array([1, 2, 3], type=pa.int64()),
                    "candidate_codebook": ["1,3", "1,3", "1,3"],
                    "priority": [1, 1, 1],
                    "score": [0.1, 0.1, 0.1],
                }
            ),
            cand_path,
        )

        args = build_parser().parse_args(
            [
                "--input_path",
                raw_path,
                "--candidate_input_path",
                cand_path,
                "--output_path",
                out_dir,
                "--writer_type",
                "ParquetWriter",
                "--max_items_per_codebook",
                "2",
            ]
        )
        stats = run_local(args)

        self.assertEqual(stats.reassigned_count, 1)
        result = parquet.read_table(os.path.join(out_dir, "part-0.parquet"))
        self.assertIn("1,2", set(result["origin_codebook"].to_pylist()))
        self.assertLessEqual(max(Counter(result["codebook"].to_pylist()).values()), 2)

    def test_local_csv_accepts_split_codes_and_compact_candidates(self) -> None:
        raw_path = os.path.join(self.test_dir, "raw_split.csv")
        cand_path = os.path.join(self.test_dir, "cand_compact.csv")
        out_dir = os.path.join(self.test_dir, "out_split")
        csv.write_csv(
            pa.table(
                {
                    "item_id": ["1", "2", "3"],
                    "code_0": ["A", "A", "A"],
                    "code_1": ["B", "B", "B"],
                }
            ),
            raw_path,
        )
        csv.write_csv(
            pa.table(
                {
                    "item_id": ["1", "2", "3"],
                    "sorted_index": ["A|C", "A|C", "A|C"],
                }
            ),
            cand_path,
        )

        args = build_parser().parse_args(
            [
                "--input_path",
                raw_path,
                "--candidate_input_path",
                cand_path,
                "--output_path",
                out_dir,
                "--reader_type",
                "CsvReader",
                "--writer_type",
                "CsvWriter",
                "--code_field",
                "",
                "--code_fields",
                "code_0,code_1",
                "--compact_candidate_field",
                "sorted_index",
                "--max_items_per_codebook",
                "2",
            ]
        )
        stats = run_local(args)

        self.assertEqual(stats.reassigned_count, 1)
        result = csv.read_csv(os.path.join(out_dir, "part-0.csv"))
        self.assertIn("A,B", set(result["origin_codebook"].to_pylist()))
        self.assertIn("A", set(result["codebook"].to_pylist()))
        self.assertLessEqual(max(Counter(result["codebook"].to_pylist()).values()), 2)

    def test_generate_odps_sql_uses_tool_capacity_and_no_random(self) -> None:
        args = build_parser().parse_args(
            [
                "--backend",
                "odps",
                "--input_path",
                "odps://proj/tables/raw_sid/ds=20260630",
                "--candidate_input_path",
                "odps://proj/tables/cand_sid/ds=20260630",
                "--output_path",
                "odps://proj/tables/final_sid/ds=20260630",
                "--max_items_per_codebook",
                "5",
                "--max_iters",
                "1",
                "--dry_run_sql",
            ]
        )
        sql = "\n".join(generate_odps_sql(args))
        self.assertIn("<= 5", sql)
        self.assertIn("PARTITION (ds='20260630')", sql)
        self.assertIn("INNER JOIN", sql)
        self.assertIn("FROM proj.raw_sid", sql)
        self.assertIn("-- final: remaining_unassigned", sql)
        self.assertNotIn("rand()", sql.lower())
        self.assertNotIn("last_layer_random", sql)

    def test_generate_odps_sql_keep_original_policy(self) -> None:
        args = build_parser().parse_args(
            [
                "--backend",
                "odps",
                "--input_path",
                "odps://proj/tables/raw_sid/ds=20260630",
                "--candidate_input_path",
                "odps://proj/tables/cand_sid/ds=20260630",
                "--output_path",
                "odps://proj/tables/final_sid/ds=20260630",
                "--max_items_per_codebook",
                "5",
                "--max_iters",
                "1",
                "--unassigned_policy",
                "keep_original",
            ]
        )
        sql = "\n".join(generate_odps_sql(args))

        self.assertIn("-- final: remaining_unassigned", sql)
        self.assertIn("u.origin_codebook AS codebook", sql)
        self.assertIn("COALESCE(cnt.cnt, 0) + u.rn AS `index`", sql)
        self.assertIn("WHERE a.item_id IS NULL AND ds='20260630'", sql)

    def test_run_odps_errors_on_remaining_unassigned(self) -> None:
        executed_sqls = []

        class FakeReader:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

            def __iter__(self):
                return iter([(1,)])

        class FakeInstance:
            def __init__(self, sql):
                self.sql = sql

            def wait_for_success(self):
                return None

            def open_reader(self):
                return FakeReader()

        class FakeODPS:
            def __init__(self, **kwargs):
                pass

            def execute_sql(self, sql):
                executed_sqls.append(sql)
                return FakeInstance(sql)

        args = build_parser().parse_args(
            [
                "--backend",
                "odps",
                "--input_path",
                "odps://proj/tables/raw_sid/ds=20260630",
                "--candidate_input_path",
                "odps://proj/tables/cand_sid/ds=20260630",
                "--output_path",
                "odps://proj/tables/final_sid/ds=20260630",
                "--max_items_per_codebook",
                "5",
                "--max_iters",
                "0",
            ]
        )

        fake_odps_module = types.SimpleNamespace(ODPS=FakeODPS)
        with mock.patch.dict(sys.modules, {"odps": fake_odps_module}):
            with mock.patch(
                "tzrec.datasets.odps_dataset._create_odps_account",
                return_value=("account", "endpoint"),
            ):
                with self.assertRaisesRegex(RuntimeError, "could not be assigned"):
                    run_odps(args)

        executed_sql = "\n".join(executed_sqls)
        self.assertIn("-- final: remaining_unassigned", executed_sql)
        self.assertNotIn("INSERT OVERWRITE TABLE proj.final_sid", executed_sql)


if __name__ == "__main__":
    unittest.main()
