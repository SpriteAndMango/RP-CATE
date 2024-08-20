# Benchmarks and Recurrent Perceptron-based Channel Attention Transformer Encoder (RP-CATE)
## Introduction
This repository contains the experiments done in the work __RP-CATE: Recurrent Perceptron-based Channel Attention Transformer Encoder for Industrial Hybrid Modeling__.  

## Framework
![RP-CATE framework](https://github.com/SpriteAndMango/RP-CATE/blob/master/RP_CATE/picture/RP-CATE.jpg)

## System Requirements
The source code developed in Python 3.8 using Tensorflow 2.12.0 and PyTorch 1.7.1. The required python dependencies are given below. RP-CATE is supported for any standard computer and operating system (Windows/macOS/Linux) with enough RAM to run. There is no additional non-standard hardware requirements.

```
tensorflow>=2.12.0
torch>=1.7.1
numpy>=1.20.2
scikit-learn>=0.24.2
pandas>=1.2.4
pandas~=1.5.1
```

## Benchmarks
The repository includes the execution of all the following benchmarks.
* Radial Basis Fucntional Neural Network (RBFNN)
* Gated Recurrent Unit (GRU)
* Long Short Term Memory (LSTM)
* Temporal Convolutional Networks (TCN)[1](https://arxiv.org/abs/1803.01271)
* Transformer[2]


## Reference

```bibtex
@article{bai2018empirical,
  title={An empirical evaluation of generic convolutional and recurrent networks for sequence modeling},
  author={Bai, Shaojie and Kolter, J Zico and Koltun, Vladlen},
  journal={arXiv preprint arXiv:1803.01271},
  year={2018}
}

@article{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish},
  journal={arXiv preprint arXiv:1706.03762},
  year={2017}
}


