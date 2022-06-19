# Repository for RepVGG pruning

This repository is a prunable implementation of [RepVGG](https://arxiv.org/abs/2101.03697). We refer to the official [RMNet repository](https://github.com/fxmeng/RMNet).

## Requirements

To install requirements:

```setup
pip install torch torchvision
```

## Training

To prune the models, run this command:

``` bash
python train_pruning.py --sr 1e-4 --threshold 5e-4 # sparse training

python train_pruning.py --eval xxx/ckpt.pth # eval

python train_pruning.py --finetune xxx/ckpt.pth # pruned finetuning
```

## TODO: Results

Our model achieves the following performance on :

## Contributing

Our code is based on [RMNet](https://github.com/fxmeng/RMNet), [RepVGG](https://github.com/DingXiaoH/RepVGG) and [nni/amc pruning](https://github.com/microsoft/nni/tree/master/examples/model_compress/pruning/amc)
