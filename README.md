# DNN Training, from theory to practice

This repository is complementary to the deep learning training class given
to les Mines PariTech on the 11th of March 2022.

You can find [here the slides of the class][slides].

## Requirements

To get started, clone it and prepare a new virtual env.

```
git clone
cd dnn_theo_practice
python3 -m venv env
source env/bin/activate
python3 -m pip install -r requirements.txt
```

**Note**: it can be safer to install PyTorch through a conda environment to
	make sure all proper versions of CUDA realted libraries are installed and used.
	We use `pip` here for simplicity.

## Basic training pipeline

To get started, you can run

```
python -m basic.train
```

You can tweak some hyper parameters:

```
python -m basic.train --lr 0.1 --epochs 30 --model mobilenet_v2
```

This basic pipeline provides all the essential tools for training a neural network:

- automatic experiment naming,
- logging and metric dumping,
- checkpointing with automatic resume.

Looking at [basic/train.py](basic/train.py) you will see that 90% of the code is
not deep learning but pure engineering.
Some frameworks like [PyTorch Lightning][pl] can
save you some of this trouble, at the cost of losing control and understanding over what happens.
In any case it is good to have an idea of how things work under the hood!

## PyTorch-Lightning training pipeline

Insite the `pl_hydra` folder, I provide the same pipeline, but using PyTorch-Lightning along with [Hydra][hydra],
as an alternative to `argparse`. Have a look at [pl_hydra/train.py](pl_hydra/train.py) to see
the differences with the previous implementation.

```
python -m pl_hydra.train optim.lr=0.1 model=mobilenet_v2
```


## Using existing frameworks:

At this point, it is a good time to introduce a few frameworks you might want to use for your projects.

### Hydra

[Hydra][hydra] handles things like logging, configuration parsing (based on YAML files, which is a bit nicer
than argparse, especially for large projects), and also has support for some grid search scheduling
with a dedicated language. It also supports meta-optimizers like Nevergrad (see after).

### Nevergrad

[Nevergrad](https://github.com/facebookresearch/nevergrad) is a framework for gradient free optimization.
It can be used to automatically tune your model or optimization hyper-parameters with smart random search.


### PyTorch-Lightning

[PyTorch Lightning][pl] takes care of logging, distributed
training, checkpointing and many more boilerplate parts of a deep learning research project.
It is powerful but also quite complex, and you will lose some control over the training pipeline.

### Dora

[Dora](https://github.com/fairinternal/dora) is an experiment management framework:
- Grid searches are expressed as pure python.
- Experiments have an automatic signature assigned based on its args.
- Keeps in sync experiments defined in grid files, and those running on the cluster.
- Basic terminal based reporting of job states, metrics etc.

Dora allows you to scale up to hundreds of experiments without losing your sanity.

### Plotting and monitoring utilities

While it is always good to have basic metric reporting inside logs, it can be
more conveniant to track experimental progress through a web browser.
[TensorBoard](https://github.com/tensorflow/tensorboard), initially developed for TensorFlow
provide just that. A fully hosted alternative is [Wandb](https://wandb.ai/).
Finally, [HiPlot](https://github.com/facebookresearch/hiplot) is a lightweight package
to easily make sense of the impact of hyperparameters on the metrics of interest.

### Unix tools

It is a good idea to learn to master the standard Unix/Linux tools!
For large scale machine learning, you will often have to run experiments on a remote cluster,
with only SSH access. `tmux` is a must have, as well as knowing at least of
one terminal based file editor (`nano` is the simplest, `emacs` or `vim` are more complex
but quite powerful). Take some time to learn about tuning your `bashrc`, setting up
aliases for often used commands etc.

You will probably need tools like `grep`, `less`, `find` or `ack`.
I personnaly really enjoy [fd](https://github.com/sharkdp/fd), an alternative to `find`
with some intuitive interface. Similarly [ag](https://github.com/ggreer/the_silver_searcher)
is a nice way to quickly look through a codebase in the terminal. If you need
to go through a lot of logs, you will enjoy [ripgreg](https://github.com/BurntSushi/ripgrep).



# License

This code in this repository is released into the public domain. You can freely reuse any part of it
and you don't even need to say where you found it! See [the LICENSE](LICENSE) for more information.

The slides are released under Creative Commons CC-BY-NC.



[pl]: https://github.com/PyTorchLightning/pytorch-lightning
[hydra]: https://github.com/facebookresearch/hydra
[dora]: https://github.com/facebookresearch/dora
[slides]: https://ai.honu.io/presentations/dnn_theo_practice.pdf
