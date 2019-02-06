# Stochastic Delta Rule

This repository holds the code for the paper 

'Dropout is a special case of the stochastic delta rule: faster and more accurate deep learning' (submitted to ICML; on [arXiv](https://arxiv.org/abs/1808.03578))

[Noah Frazier-Logue](https://www.linkedin.com/in/noah-frazier-logue-1524b796/), [Stephen Jose Hanson](http://nwkpsych.rutgers.edu/~jose/)

Stochastic Delta Rule (SDR) is a weight update mechanism that assigns to each weight a standard deviation that changes as a function of the gradients every training iteration. At the beginning of each training iteration, the weights are re-initialized using a normal distribution bound by their standard deviations. Over the course of the training iterations and epochs, the standard deviations converge towards zero as the network becomes more sure of what the values of each of the weights should be. For a more detailed description of the method and its properties, [have a look at the paper](https://arxiv.org/abs/1808.03578).


## Results

[Here is a TensorBoard instance](https://boards.aughie.org/board/EchkCFmhLRg4tzFlcff5DUMX4i0/#scalars&_smoothingWeight=0) that shows the results from the paper regarding titration of training epochs and the comparison to dropout (on DN100/CIFAR-100).

### Dropout

|Model type            |Depth  |C10    |C100   |
|:---------------------|:------|:------|:------|
|DenseNet(*k* = 12)    |40     |6.88   |27.88  |
|DenseNet(*k* = 12)    |100    |----   |24.67  |
|DenseNet-BC(*k* = 12) |250    |----   |23.91  |

### SDR

|Model type            |Depth  |C10    |C100   |
|:---------------------|:------|:------|:------|
|DenseNet(*k* = 12)    |40     |**5.95**   |**25.25**  |
|DenseNet(*k* = 12)    |100    |----   |**21.72**  |
|DenseNet-BC(*k* = 12) |250    |----   |**19.79**  |


Two types of [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993) (DenseNets) are available:

- DenseNet - without bottleneck layers
- DenseNet-BC - with bottleneck layers

Each model can be tested on such datasets:

- CIFAR-10
- CIFAR-100
- ImageNet (results coming soon)

A number of layers, blocks, growth rate, image normalization and other training params may be changed trough shell or inside the source code.

## Usage

Example run:

```
    python train.py --layers 40 --no-bottleneck --growth 12 --reduce 1.0 -b 100 --epochs 100 --name DN40_C100_alpha_0.25_beta_0.05_zeta_0.7 --tensorboard --sdr --dataset C100 --lr 0.25 --beta 0.52 --zeta 0.7
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


## TensorBoard logs and steps to reproduce results

### DenseNet-40 on CIFAR-10

[TensorBoard Log](https://boards.aughie.org/board/LMcrxHaX-ahRA_hCMGjSxE-0huY/#scalars&_smoothingWeight=0)

Command:
```
    python train.py --layers 40 --no-bottleneck --growth 12 --reduce 1.0 -b 100 --epochs 100 --name DN40_C10_alpha_0.25_beta_0.1_zeta_0.999 --tensorboard --sdr --dataset C10 --lr 0.25 --beta 0.1 --zeta 0.999
```

### DenseNet-40 on CIFAR-100

[TensorBoard Log](https://boards.aughie.org/board/GNcmrOhQdxgwQXx2rppuQWmPSf0/#scalars&_smoothingWeight=0)

Command:
```
    python train.py --layers 40 --no-bottleneck --growth 12 --reduce 1.0 -b 100 --epochs 100 --name DN40_C100_alpha_0.25_beta_0.05_zeta_0.7 --tensorboard --sdr --dataset C100 --lr 0.25 --beta 0.05 --zeta 0.7
```

### DenseNet-100 on CIFAR-100

[TensorBoard Log](https://boards.aughie.org/board/0L-rz-a7b_L51jg26kPUCX59yJM/#scalars&_smoothingWeight=0)

Command:
```
    python train.py --layers 100 --no-bottleneck --growth 12 --reduce 1.0 -b 100 --epochs 100 --name DN100_C100_alpha_0.25_beta_0.1_zeta_0.7 --tensorboard --sdr --dataset C100 --lr 0.25 --beta 0.1 --zeta 0.7
```

### DenseNet-250 BC on CIFAR-100

[TensorBoard Log](https://boards.aughie.org/board/FbVdH33aGV50OeW49LgFRDK96D8/#scalars&_smoothingWeight=0)

Command:
```
    python train.py --layers 250 --growth 12 --reduce 1.0 -b 100 --epochs 100 --name DN250_C100_alpha_0.25_beta_0.03_zeta_0.5 --tensorboard --sdr --dataset C100 --lr 0.25 --beta 0.03 --zeta 0.5
```


The code used is based heavily on [Andreas Veit's DenseNet implementation](https://github.com/andreasveit/densenet-pytorch) and [PyTorch's Vision repository](https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py).


### Dependencies
* [PyTorch](http://pytorch.org/)
* [NumPy](https://www.numpy.org/)

#### Optional
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


