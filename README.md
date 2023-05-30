# CS503_project

Authors: Sebastián Breguel Gonzalez, Aitor Ganuza Izagirre

This is the repository for the CS 503 Visual intelligence: Machines and Minds project.

You can install the requirements by running `pip install -r requirements.txt`.

```bash

main.py
|
|__📜main.py # data class
|__📜dataset.py # data class
|__📜models.py # model class
|__📜utils.py  # utils function to create, get, process and run data/metrics/models
|
|__📂Layers
|   |__📜patch_embeddings.py
|   |__📜positional_encodings.py
|   |__📜helper.py
|   |__📂transformers.py
|__📂yamls  #Configurations to run the models


```

## Datasets

here is the list of datasets with the subset of variations we are going to use:

- MNIST
- CIFAR10: 60K images, 10 classes
  - CIFAR10-C
- CIFAR100: 60K images, **100** classes
  - CIFAR100-C
- FOOD101: 101K images, 101 classes
  - FOOD101-C: we to corrupt the images

### To run the code

You should have an yamls file with the configuration of the model you want to run. running the command:
`python main.py --config <path_to_yamls_file>`

If you dont provide a path to the yamls file, it will run the default the ViT model.

## References

Attentions:

- [Axial-DeepLab: Stand-Alone Axial-Attention for Panoptic Segmentation](https://arxiv.org/pdf/2003.07853.pdf)
- [Castling Vit](https://arxiv.org/pdf/2211.10526.pdf)
- [CeiT:Incorporating Convolution Designs into Visual Transformers](https://arxiv.org/abs/2103.11816)
- [CvT: Introducing Convolutions to Vision Transformers](https://arxiv.org/pdf/2103.15808.pdf)
- [Self-Attention with Relative Position Representations](https://arxiv.org/pdf/1803.02155v2.pdf)
- [Primer: Searching for Efficient Transformers for Language Modeling](https://arxiv.org/pdf/2109.08668.pdf) (used on NLP)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/pdf/2104.09864.pdf) (used on NLP)
- [Towards Robust Vision Transformers](https://arxiv.org/pdf/2105.07926.pdf)
- [Train Short, Test Long: Attention With Linear Biases Enables Input Length Extrapolation](https://arxiv.org/pdf/2108.12409.pdf) (used on NLP)

Robustness:

- [MedViT: A Robust Vision Transformer for Generalized Medical Image Classification](https://arxiv.org/abs/2302.09462)
- [Robust Transformer with Locality Inductive Bias and Feature Normalization](https://arxiv.org/pdf/2301.11553.pdf)

Others:

- [Attention is all You need](https://arxiv.org/pdf/1706.03762.pdf)
- [Early Convolutions Help Transformers See Better](https://arxiv.org/pdf/2106.14881v2.pdf)
- [Vision transformer](https://arxiv.org/pdf/2010.11929.pdf)
- [ResT: An Efficient Transformer for Visual Recognition](https://arxiv.org/pdf/2105.13677.pdf)
- [Three things everyone should know about Vision Transformers](https://arxiv.org/pdf/2203.09795.pdf)
