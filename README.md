My own implementation of the algorithm described in [Your Classifier is Secretly an Energy Based Model and You Should Treat it Like One](https://arxiv.org/abs/1912.03263)

Code of original authors of paper can be found at
(https://github.com/wgrathwohl/JEM)

Includes a script to train some basic models on both MNIST and FMNIST, with output viewable in Tensorboard.

Defaults should run fine, see the arguments list on how to modify.

Note: BatchNorm doesn't seem to work. The classification works OK, but generated images seem to stay close to random noise.
