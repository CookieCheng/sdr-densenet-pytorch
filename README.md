# Stochastic Delta Rule

This repository holds the code for the paper 

'Dropout is a special case of the stochastic delta rule: faster and more accurate deep learning' (submitted to ICML; on [arXiv](https://arxiv.org/abs/1808.03578))

[Noah Frazier-Logue](https://www.linkedin.com/in/noah-frazier-logue-1524b796/), [Stephen Jose Hanson](http://nwkpsych.rutgers.edu/~jose/)

Stochastic Delta Rule (SDR) is a weight update mechanism that assigns to each weight a standard deviation that changes as a function of the gradients every training iteration. At the beginning of each training iteration, the weights are re-initialized using a normal distribution bound by their standard deviations. Over the course of the training iterations and epochs, the standard deviations converge towards zero as the network becomes more sure of what the values of each of the weights should be. For a more detailed description of the method and its properties, have a look at the paper [link here].



Two types of [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993) (DenseNets) are available:

- DenseNet - without bottleneck layers
- DenseNet-BC - with bottleneck layers

Each model can be tested on such datasets:

- CIFAR-10
- CIFAR-10+ (with data augmentation)
- CIFAR-100
- CIFAR-100+ (with data augmentation)
- ImageNet (coming soon)

A number of layers, blocks, growth rate, image normalization and other training params may be changed trough shell or inside the source code.

## Usage

Example run:

```
    python train.py --layers 40 --no-bottleneck --growth 12 --reduce 1.0 -b 100 --epochs 100 --name DN40_C100_alpha_0.25_beta_0.02_zeta_0.7 --tensorboard --sdr --dataset C100 --lr 0.25 --beta 0.02 --zeta 0.7
```

This run would train a 40-layer DenseNet model on CIFAR-100 and log the progress to TensorBoard. To use dropout, run something like

```
    python train.py --layers 40 --no-bottleneck --growth 12 --reduce 1.0 -b 100 --epochs 100 --name DN40_C100_do_0.2 --tensorboard --dataset C100 --droprate 0.2
```

where `--droprate` is the probability (in this case 20%) that a neuron is dropped during dropout.

**NOTE:** the `--sdr` argument will override the `--droprate` argument. For example:

```
    python train.py --layers 40 --no-bottleneck --growth 12 --reduce 1.0 -b 100 --epochs 100 --name DN40_C100_alpha_0.25_beta_0.02_zeta_0.7 --tensorboard --sdr --dataset C100 --lr 0.25 --beta 0.02 --zeta 0.7 --droprate 0.2
```

will use SDR and not dropout.


List all available options:

```    
    python train.py --help
```

Below is the description of DenseNet from the [repository on which this was heavily based](https://github.com/andreasveit/densenet-pytorch), with appropriate modifications:

## DenseNets
[DenseNets [1]](https://arxiv.org/abs/1608.06993) were introduced in late 2016 after to the discoveries by [[2]](https://arxiv.org/abs/1603.09382) and [[3]](https://arxiv.org/abs/1605.06431) that [residual networks [4]](https://arxiv.org/abs/1512.03385) exhibit extreme parameter redundancy. DenseNets address this shortcoming by reducing the size of the modules and by introducing more connections between layers. In fact, the output of each layer flows directly as input to all subsequent layers of the same feature dimension as illustrated in their Figure 1 (below). This increases the dependency between the layers and thus reduces redundancy.

<img src="https://github.com/andreasveit/densenet-pytorch/blob/master/images/Fig1.png?raw=true" width="400">

The improvements in accuracy per parameter are illustrated in their results on ImageNet (Figure 3). 

<img src="https://github.com/andreasveit/densenet-pytorch/blob/master/images/FIg3.png?raw=true" width="400">

## This implementation
This implementation is quite _memory efficient requiring between 10% and 20% less memory_ compared to the original torch implementation. We optain a final test error of 4.76 % with DenseNet-BC-100-12 (paper reports 4.51 %) and 5.35 % with DenseNet-40-12 (paper reports 5.24 %).

This implementation allows for __all model variants__ in the DenseNet paper, i.e., with and without bottleneck, channel reduction, data augmentation and dropout. 

For simple configuration of the model, this repo uses `argparse` so that key hyperparameters can be easily changed.

Further, this implementation supports [easy checkpointing](https://github.com/andreasveit/densenet-pytorch/blob/master/train.py#L136), keeping track of the best model and [resuming](https://github.com/andreasveit/densenet-pytorch/blob/master/train.py#L103) training from previous checkpoints.

### Tracking training progress with TensorBoard
To track training progress, this implementation uses [TensorBoard](https://www.tensorflow.org/guide/summaries_and_tensorboard) which offers great ways to track and compare multiple experiments. To track PyTorch experiments in TensorBoard we use [tensorboardX](https://github.com/lanpa/tensorboardX) which can be installed with 
```
pip install tensorboardx
```
Example training curves for DenseNet-BC-100-12 (dark blue) and DenseNet-40-12 (light blue) for training loss and validation accuracy is shown below. 

![Training Curves](images/Fig4.png)

### Dependencies
* [PyTorch](http://pytorch.org/)
* [NumPy](https://www.numpy.org/)

optional:
* [tensorboardX](https://github.com/lanpa/tensorboardX)


### Cite
If you use DenseNets in your work, please cite the original paper as:
```
@article{Huang2016Densely,
  author  = {Huang, Gao and Liu, Zhuang and Weinberger, Kilian Q.},
  title   = {Densely Connected Convolutional Networks},
  journal = {arXiv preprint arXiv:1608.06993},
  year    = {2016}
}
```

If this implementation is useful to you and your project, please also consider to cite or acknowledge this code repository.

### References 
[1] Huang, G., Liu, Z., Weinberger, K. Q., & van der Maaten, L. (2016). Densely connected convolutional networks. arXiv preprint arXiv:1608.06993.

[2] Huang, G., Sun, Y., Liu, Z., Sedra, D., & Weinberger, K. Q. (2016). Deep networks with stochastic depth. In European Conference on Computer Vision (ECCV '16)

[3] Veit, A., Wilber, M. J., & Belongie, S. (2016). Residual networks behave like ensembles of relatively shallow networks. In Advances in Neural Information Processing Systems (NIPS '16)

[4] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Conference on Computer Vision and Pattern Recognition (CVPR '16)

