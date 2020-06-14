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
from time import time
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import logging
import argparse
from pathlib import Path
from typing import Tuple, List

from models import BasicNet, DeeperNet
from loaders import loader_mnist, loader_fmnist
from samplebuffer import SampleBuffer
from evaluate import evaluate_acc, generate_unconditional


RANDOM_SEED = 42


torch.manual_seed(RANDOM_SEED)


def sgld(
    model: nn.Module,
    buffer: SampleBuffer,
    nr_samples: int,
    pcd_length: int,
    alpha: float,
    sigma: float,
    random_reinit_freq: float,
) -> Tuple[torch.Tensor, List]:
    samples, classes = buffer.sample(nr_samples, random_reinit_freq)
    samples = samples[torch.randperm(nr_samples)]
    model.eval()
    samples = samples.to(device)
    energies = []
    x_t = samples.detach().requires_grad_(True)
    for t in range(pcd_length):
        energy = -model(x_t).logsumexp(dim=1).sum()
        x_t.data += -alpha * torch.autograd.grad(energy, x_t)[
            0
        ] + sigma * torch.randn_like(x_t)
        # TODO: somehow normalize based on this data?
        energies.append(energy.detach().cpu().numpy())
    samples = x_t.detach()
    model.train()
    buffer.push(samples, classes)
    return samples, energies


def save_checkpoint(model: nn.Module, buffer: SampleBuffer, save_dir: Path, tag):
    checkpoint_dict = {"model_state": model.state_dict(), "sample_buffer": buffer}
    torch.save(checkpoint_dict, save_dir / tag)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=True, type=bool)
    parser.add_argument("--dataset", default="MNIST", choices=["MNIST", "FMNIST"])
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--n_epochs", default=20, type=int)
    parser.add_argument("--chain_length", default=20, type=int)
    parser.add_argument("--reinit_rate", default=0.05, type=float)
    parser.add_argument("--alpha", default=1.0, type=float)
    parser.add_argument("--sigma", default=0.01, type=float)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--buffer_size", default=1e4, type=int)
    parser.add_argument("--image_noise", default=0.01, type=float)
    parser.add_argument("--save_dir", default=".", type=str)

    args = parser.parse_args()

    REINIT_FREQ_START = 0.05
    REINIT_DECAY_RATE = 1.0
    REINIT_MIN = 0.05

    if args.dataset == "MNIST":
        train_loader = loader_mnist(args.batch_size, image_noise=args.image_noise)
        test_loader = loader_mnist(args.batch_size, train=False)
    elif args.dataset == "FMNIST":
        train_loader = loader_fmnist(args.batch_size, image_noise=args.image_noise)
        test_loader = loader_fmnist(args.batch_size, train=False)
    reinit_freq = args.reinit_rate

    if args.cuda:
        device = torch.device("cuda")
        torch.backends.cudnn.enabled = True
    else:
        device = torch.device("cpu")

    image_shape = iter(train_loader).next()[0].shape[1:]

    network = BasicNet(input_shape=image_shape)
    network.to(device)

    optimizer = optim.SGD(network.parameters(), lr=args.lr)
    samplebuffer = SampleBuffer(max_samples=args.buffer_size)
    objective_clf = nn.CrossEntropyLoss()
    writer = SummaryWriter()

    for e in range(args.n_epochs):
        time0 = time()
        reinit_freq = max(REINIT_MIN, reinit_freq * REINIT_DECAY_RATE)
        losses_clf = []
        losses_gen = []
        energy_list = []
        for images, labels in tqdm(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            samples, energies = sgld(
                network,
                samplebuffer,
                images.shape[0],
                args.chain_length,
                args.alpha,
                args.sigma,
                reinit_freq,
            )
            output = network(images)
            loss_clf = objective_clf(output, labels)
            loss_gen = -torch.mean(
                torch.logsumexp(output, dim=1)
                - torch.logsumexp(network(samples), dim=1)
            )
            loss_total = loss_clf + loss_gen
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
            losses_clf.append(loss_clf.detach().cpu().numpy())
            losses_gen.append(loss_gen.detach().cpu().numpy())
            energy_list.append(energies)
        else:
            acc = evaluate_acc(test_loader, network)
            logging.info(
                f"Epoch: {e}, classification accuracy: {acc}, time taken {time() - time0:.2f}, reinit freq: {reinit_freq}, generator loss: {np.mean(losses_gen):.2f}, classifier loss: {np.mean(losses_clf):.2f}, mean energy: {np.mean(energies):.2f}."
            )
            writer.add_scalar("accuracy", acc, e)
            writer.add_scalar("generator_loss", np.mean(losses_gen), e)
            writer.add_scalar("classifier_loss", np.mean(losses_clf), e)
            writer.add_scalar("energy", np.mean(energies), e)
            examples, _ = samplebuffer.sample(5, reinit_freq=0)
            writer.add_image("generated", vutils.make_grid(examples.data), e)
            writer.add_image("real", vutils.make_grid(images.data[:5]), e)
            save_checkpoint(
                network, samplebuffer, Path(args.save_dir), "last_checkpoint.pt"
            )
