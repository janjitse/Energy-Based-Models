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

import torch
import torchvision


def loader_mnist(
    batch_size: int, train: bool = True, image_noise=0.01
) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            "files/",
            train=train,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.5,), (0.5,)),
                    lambda x: x + image_noise * torch.randn_like(x) if train else x,
                ]
            ),
        ),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
    )


def loader_fmnist(
    batch_size: int, train: bool = True, image_noise=0.01
) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        torchvision.datasets.FashionMNIST(
            "files/",
            train=train,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.5,), (0.5,)),
                    lambda x: x + image_noise * torch.randn_like(x) if train else x,
                ]
            ),
        ),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
    )
