# Copyright 2020 Jan Jitse Venselaar.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import torch
import numpy as np
from typing import NoReturn, Tuple


class SampleBuffer:
    def __init__(self, max_samples: int = 10e4):
        self.max_samples = max_samples
        self.buffer = []

    def __len__(self):
        return len(self.buffer)

    def push(self, samples: torch.tensor, class_ids: torch.tensor) -> NoReturn:
        samples = samples.detach().to("cpu")
        class_ids = class_ids.detach().to("cpu")
        for sample, class_id in zip(samples, class_ids):
            self.buffer.append((sample, class_id))
            if len(self.buffer) > self.max_samples:
                self.buffer.pop(0)

    def get(self, n_samples: int) -> Tuple[torch.tensor, torch.tensor]:
        items = random.choices(self.buffer, k=n_samples)
        samples, class_ids = zip(*items)
        samples = torch.stack(samples, 0)
        class_ids = torch.stack(class_ids, 0)
        return samples, class_ids

    def sample(
        self, batch_size: int, reinit_freq: float = 0.05
    ) -> Tuple[torch.tensor, torch.tensor]:

        reinits = sum(np.random.rand(batch_size) < reinit_freq)
        if batch_size > reinits and len(self.buffer) > 0:
            replay_samples, replay_class_ids = self.get(batch_size - reinits)
        else:
            replay_samples = torch.empty(0, 1, 28, 28)  # , requires_grad=True)
            replay_class_ids = torch.randint(0, 10, (0,))
        random_sample_size = batch_size - replay_samples.shape[0]
        random_samples = torch.rand(random_sample_size, 1, 28, 28) * 2 - 1
        random_class_ids = torch.randint(0, 10, (random_sample_size,))
        return (
            torch.cat([replay_samples, random_samples]),
            torch.cat([replay_class_ids, random_class_ids]),
        )
