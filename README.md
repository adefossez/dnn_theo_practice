# DNN Training, from theory to practice

This repository is complementary to the deep learning training class given
to les Mines PariTech on the 11th of March 2022.

You can find [here the slides of the class][slides].

This repository contains ...

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
as an alternative to `argparse`.

## Dora based training pipeline

Finally, `dora_train` contains the same pipeline using both PyTorch-Lightning, Hydra and [Dora][dora].

## Using existing frameworks:

At this point, it is a good time to introduce a few frameworks you might want to use for your project,
replacing or completing what we have done so far.

### Stool

For simple grid search cases, I recommend the [stool](https://github.com/fairinternal/stool)
tool. In one line of shell, you can schedule remote jobs, and perform grid searches
based on a json specification. Unlike the solution we developed in `schedule.py`,
`stool` does not ensure that a single job is assigned to a given experiment, and one must be careful
not to schedule multiple time an experiment.

After having followed the installation procedure, you could easily schedule a remote job with
```
stool run ./train.py
```

### Hydra

[Hydra]() handles things like logging, configuration parsing (based on YAML files, which is a bit nicer
than argparse, especially for large projects), and also has support for some grid search scheduling
with a dedicated language. It also supports meta-optimizers like Nevergrad (see after).

### Nevergrad

[Nevergrad](https://github.com/facebookresearch/nevergrad) is a framework for gradient free optimization.
It can be used to automatically tune your model or optimization hyper-parameters with smart random search.


### PyTorch-Lightning

[PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) takes care of logging, distributed
training, checkpointing and many more boilerplate parts of a deep learning research project.
It is powerful but also quite complex, and you will lose some control over the training pipeline.

### Dora

[Dora](https://github.com/fairinternal/dora) is an experiment management framework.
The core concepts are similar to what we defined in `schedule.py` and `grid.py`:
- Grid searches are expressed as pure python.
- Experiments have an automatic signature assigned based on its args.
- Keeps in sync experiments defined in grid files, and those running on the cluster.
- Basic terminal based reporting of job states, metrics etc.

Dora allows you to scale up to dozens of grid searches in parallel with hundreds of
experiments without losing your sanity.

### HiPlot


[HiPlot](https://github.com/facebookresearch/hiplot) or

### Tensorboard

	[TensorBoard](https://github.com/tensorflow/tensorboard)


# License

This repository is released into the public domain. You can freely reuse any part of it
and you don't even need to say where you found it! See [the LICENSE](LICENSE) for more information.

pl: https://github.com/PyTorchLightning/pytorch-lightning
hydra: https://github.com/facebookresearch/hydra
dora: https://github.com/facebookresearch/dora
