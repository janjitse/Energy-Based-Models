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

import matplotlib.pyplot as plt
import numpy as np


def view_classify(image, probabilities):
    ps = probabilities.data.numpy().squeeze()
    fig, ax = plt.subplots(1, 2)
    ax[0].axis("off")
    ax[0].imshow(image.data.numpy().squeeze())
    ax[0].set_title(f"Predicted Digit = {ps.argmax()}")
    ax[1].bar(np.arange(10), ps)
    ax[1].set_title("Class probability")
    ax[1].set_xticks(np.arange(10))
    ax[1].set_ylim(0, 1.1)
