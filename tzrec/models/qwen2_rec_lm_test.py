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

import types
import unittest

import torch
from torch import nn

from tzrec.models.qwen2_rec_lm import Qwen2RecLM


def _stub(num_levels=3, base_vocab=100, pad_id=9, device="cpu"):
    """A Qwen2RecLM with the splice-relevant state wired up, no HF backbone.

    Template buffers use tiny placeholder ids so the spliced layout is easy to
    read; real buffers come from ``_build_prompt_tokens`` at init time.
    """
    m = object.__new__(Qwen2RecLM)
    nn.Module.__init__(m)
    m._ignore_index = -100
    m._num_levels = num_levels
    m._base_vocab = base_vocab
    m._pad_token_id = pad_id
    m.lm = types.SimpleNamespace(device=torch.device(device))
    for name, vals in {
        "tpl_system": [10, 11], "tpl_user_prefix": [12], "tpl_user_suffix": [13],
        "tpl_asst_prefix": [14], "tpl_asst_suffix": [15], "tpl_eos": [9],
    }.items():
        m.register_buffer(name, torch.tensor(vals, dtype=torch.long), persistent=False)
    return m


class Qwen2RecLMTest(unittest.TestCase):
    def test_splice_layout_and_labels(self) -> None:
        m = _stub()
        u = [torch.tensor([100, 101, 102])]
        a = [torch.tensor([200, 201, 202])]  # 3 codes = num_levels
        ids, labels, mask = m._splice_input_ids(u, a)
        # [system | user_prefix | history | user_suffix |
        #  asst_prefix | answer | asst_suffix | eos]
        self.assertEqual(
            ids[0].tolist(), [10, 11, 12, 100, 101, 102, 13, 14, 200, 201, 202, 15, 9]
        )
        # only the answer (cols 8-10) and the trailing eos (col 12) are supervised
        self.assertEqual(
            labels[0].tolist(),
            [-100] * 8 + [200, 201, 202, -100, 9],
        )
        self.assertEqual(mask[0].tolist(), [1] * 13)

    def test_left_padding_varied_lengths(self) -> None:
        m = _stub()
        u = [torch.tensor([100, 101, 102, 103]), torch.tensor([100])]
        a = [torch.tensor([200, 201, 202]), torch.tensor([207, 208, 209])]
        ids, labels, mask = m._splice_input_ids(u, a)
        T = ids.shape[1]
        n1 = 2 + 1 + 1 + 1 + 1 + 3 + 1 + 1  # shorter row's real length
        # shorter row is left-padded: pad at the front, content right-aligned
        self.assertEqual(ids[1, : T - n1].tolist(), [m._pad_token_id] * (T - n1))
        self.assertEqual(mask[1].tolist(), [0] * (T - n1) + [1] * n1)
        self.assertEqual(labels[1, : T - n1].tolist(), [-100] * (T - n1))
        # every row's trailing eos is supervised and the answer ends just before
        self.assertEqual(labels[:, -1].tolist(), [9, 9])

    def test_mask_keeps_trailing_eos_when_pad_equals_eos(self) -> None:
        # pad_id == eos value: the mask must NOT mask the real trailing eos
        m = _stub(pad_id=9)  # tpl_eos == 9 too
        ids, _, mask = m._splice_input_ids(
            [torch.tensor([100])], [torch.tensor([200, 201, 202])]
        )
        self.assertEqual(int(ids[0, -1]), 9)
        self.assertEqual(int(mask[0, -1]), 1)
        self.assertEqual(mask[0].tolist(), [1] * ids.shape[1])

    def test_min_first_non_neg_index(self) -> None:
        labels = torch.tensor([[-100, -100, 5, 6], [-100, 7, 8, 9]])
        self.assertEqual(Qwen2RecLM._min_first_non_neg_index(labels), 1)

    def test_build_prompt_tokens_registers_buffers(self) -> None:
        m = object.__new__(Qwen2RecLM)
        nn.Module.__init__(m)
        tok = types.SimpleNamespace(
            eos_token_id=99,
            encode=lambda text, add_special_tokens=False: [len(text)],
        )
        cfg = types.SimpleNamespace(
            system_instruction="", user_prefix_text="", user_suffix_text=""
        )
        m._build_prompt_tokens(tok, cfg)
        for name in [
            "tpl_system", "tpl_user_prefix", "tpl_user_suffix",
            "tpl_asst_prefix", "tpl_asst_suffix", "tpl_eos",
        ]:
            buf = getattr(m, name)
            self.assertIsInstance(buf, torch.Tensor)
            self.assertEqual(buf.dtype, torch.int64)
        self.assertEqual(m.tpl_eos.tolist(), [99])  # eos cached for supervision


if __name__ == "__main__":
    unittest.main()
