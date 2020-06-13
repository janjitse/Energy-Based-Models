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

import torch.nn as nn


class BasicNet(nn.Module):
    def __init__(self, input_shape):
        super(BasicNet, self).__init__()
        self.conv_pipe = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=3),  # 26
            nn.ReLU(),
            # nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),  # 24
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 13
            # nn.Dropout2d(0.25),
            nn.Flatten(),
            # nn.BatchNorm1d(9216),
            nn.Linear(9216, 128),  # 64*13*13
            # nn.Dropout(0.5),
            nn.ReLU(),
            # nn.BatchNorm1d(128),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.conv_pipe(x)


class DeeperNet(nn.Module):
    def __init__(self, input_shape):
        super(DeeperNet, self).__init__()
        self.conv_pipe = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=3),  # 26
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3),  # 24
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),  # 24
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(p=0.4),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),  # 22
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),  # 20
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2),  # 20
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(p=0.4),
            nn.Flatten(),
            nn.Linear(64 * 20 * 20, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.BatchNorm1d(128),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.conv_pipe(x)
