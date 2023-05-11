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

- [MedVit](https://github.com/Omid-Nejati/MedViT/blob/main/MedViT.py)
- [Robust Transformer with Locality Inductive Bias and Feature Normalization](https://github.com/Omid-Nejati/Locality-iN-Locality)
- [Castling Vit](https://arxiv.org/pdf/2211.10526.pdf)
- Making Vision Transformers Efficient from A Token Sparsification View

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

- [ViT code](https://github.com/google-research/vision_transformer)/[Paper](https://arxiv.org/pdf/2010.11929.pdf)
- [CvT: Introducing Convolutions to Vision Transformers](https://arxiv.org/pdf/2103.15808.pdf)
- [Three things everyone should know about Vision Transformers](https://arxiv.org/pdf/2203.09795.pdf)
- [Early Convolutions Help Transformers See Better](https://arxiv.org/pdf/2106.14881.pdf)
- [Towards Robust Vision Transformers](https://arxiv.org/pdf/2105.07926.pdf)

NLP field:

- [Primer: Searching for Efficient Transformers for Language Modeling](https://arxiv.org/pdf/2109.08668.pdf)
