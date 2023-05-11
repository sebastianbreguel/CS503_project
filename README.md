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
|   |__📜patch_embeddings.py # model class
|   |__📜positional_encodings.py # model class
|   |__📜transformers.py # model class

```

## Usage

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
    - ResNet

--optimizer: Optimizer to train, could be SGD or Adam
        - Adam
        - SGD
        - AdamW
```

#### Examples

1: Run model Vit with Adam optimizer and CrossEntropyLoss loss on MNIST dataset for 2 epochs

```python
python main.py --model ViT --optimizer Adam --loss CE --dataset MNIST --epoch 2
```

## References

- ViT/[code](https://github.com/google-research/vision_transformer): [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/pdf/2010.11929.pdf)
- [CvT: Introducing Convolutions to Vision Transformers](https://arxiv.org/pdf/2103.15808.pdf)
- [Three things everyone should know about Vision Transformers](https://arxiv.org/pdf/2203.09795.pdf)
