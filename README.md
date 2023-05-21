# CS503_project

Authors: Sebastián Breguel Gonzalez, Aitor Ganuza Izagirre

This is the repository for the CS 503 Visual intelligence: Machines and Minds project.

You can install the requirements by running `pip install -r requirements.txt`.

```bash
main.py
|
|__📜utils.py  # utils function to create, get, process and run data/metrics/models
|__📜models.py # model class
|__📜dataset.py # data class
|__📜main.py # data class
|
|__📂Layers
|   |__📜patch_embeddings.py
|   |__📜positional_encodings.py
|   |__📜transformers.py
|__📂yamls  #Configurations to run the models


```

## Datasets

here is the list of datasets with the subset of variations we are going to use:

- MNIST
- CIFAR10: 60K images, 10 classes
  - CIFAR10-C
- CIFAR100: 60K images, **100** classes
  - CIFAR100-C
- IMAGENET
  - [ImageNet](http://www.image-net.org/): 1.2 million images, 1000 classes
  - [ImageNet-21K](https://patrykchrabaszcz.github.io/Imagenet32/): 14 million images, 21K classes
  - ImageNet-C: Common corruptions
  - ImageNet-P: Common perturbations
  - ImageNet-R: Sematinc Shifts
  - ImageNet-O: Out of domain distributions
  - ImageNet-A: Adversarial examples
  - ImageNet-9: Background dependence
  - ImageNet-Sketch: pen

## TODO

- [ ] Agregate ImageNet dataset
- [ ] Define metrics and datasets corrupted
- [ ] Add functions to load and save models

- [ ] Define final models and architectures
- [ ] Generate general yamls models and test
- [ ] Make pipeline for running models

Review(posible implementation):

Review for implementation and future ideas:

- [Learning a Fourier Transform for Linear Relative Positional Encodings in Transformers](https://paperswithcode.com/paper/learning-a-fourier-transform-for-linear)

### Arguments to run model

```bash
--dataset: determinate the large of the prefix
        - MNIST
        - CIFAR10
        - CIFAR100
        - IMAGENET

--loss: Loss to train,
        - CrossEntropyLoss: CE
        - NLLLoss: NLL

--model: Model to train,
    - ViT

--optimizer: Optimizer to train, could be SGD or Adam
        - Adam
        - SGD
        - AdamW
```

#### Examples

```python
python main.py
```

## References

- [CvT: Introducing Convolutions to Vision Transformers](https://arxiv.org/pdf/2103.15808.pdf)
- [Castling Vit](https://arxiv.org/pdf/2211.10526.pdf)
- [Robust Transformer with Locality Inductive Bias and Feature Normalization](https://arxiv.org/pdf/2301.11553.pdf)
- [Axial-DeepLab: Stand-Alone Axial-Attention for Panoptic Segmentation](https://arxiv.org/pdf/2003.07853.pdf)

- [TRAIN SHORT, TEST LONG: ATTENTION WITH LINEAR BIASES ENABLES INPUT LENGTH EXTRAPOLATION](https://arxiv.org/pdf/2108.12409.pdf)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/pdf/2104.09864.pdf)

- [Towards Robust Vision Transformers](https://arxiv.org/pdf/2105.07926.pdf)
- [ResT: An Efficient Transformer for Visual Recognition](https://arxiv.org/pdf/2105.13677.pdf)
- [Early Convolutions Help Transformers See Better](https://arxiv.org/pdf/2106.14881v2.pdf)
- [CeiT](https://arxiv.org/abs/2103.11816)
- [Attention is all You need](https://arxiv.org/pdf/1706.03762.pdf)
- [Self-Attention with Relative Position Representations](https://arxiv.org/pdf/1803.02155v2.pdf)
- [Original Vision Transformer Model](https://arxiv.org/pdf/2010.11929.pdf)
- [Three things everyone should know about Vision Transformers](https://arxiv.org/pdf/2203.09795.pdf)
- [MedViT: A Robust Vision Transformer for Generalized Medical Image Classification](https://arxiv.org/abs/2302.09462)
-
