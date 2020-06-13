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
import torch.nn.functional as F
import matplotlib.pyplot as plt
from samplebuffer import SampleBuffer

device = torch.device("cuda")


def evaluate_acc(
    test_loader: torch.utils.data.DataLoader, model: torch.nn.Module
) -> float:
    correct = 0
    all_count = 0
    model.eval()
    for images, labels in test_loader:
        images = images.to(device)
        # labels = labels.to(device)
        with torch.no_grad():
            outputs = model(images)
        ps = F.softmax(outputs, dim=0)
        pred_labels = ps.cpu().numpy().squeeze().argmax(axis=1)
        true_labels = labels.numpy().squeeze()
        correct += sum(pred_labels == true_labels)
        all_count += true_labels.shape[0]
    model.train()
    return correct / all_count


def generate_unconditional(buffer: SampleBuffer, amount=5):
    examples, _ = buffer.sample(amount, reinit_freq=0)
    fig, ax = plt.subplots(1, amount)
    for i in range(amount):
        ax[i].axis("off")
        ax[i].imshow(examples[i].data.numpy().squeeze())
    return fig
