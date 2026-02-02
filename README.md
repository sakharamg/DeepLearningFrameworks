# DeepLearningFrameworks

Hands-on labs and slides for a Deep Learning Frameworks course covering PyTorch and TensorFlow.

## Contents 
- Session 1: Intro to Pytorch, Pytorch datatypes and operations, Linear Regressionr
- Session 2: Non-linear boundaries, activation functions, backprop, autograd, dataset, dataloader, regularization
- Session 3: CNNs, batch normalization, saliency maps, activation maps, exploding/vanishing gradients, gradient clipping
- Session 4: Pretrained vision models (VGG, ResNet), residual connections, transfer learning, finetuning, object detection, LR scheduler, gradient accumulation
- Session 5: Segmentation (UNet, DeepLabv3), mixed precision training, autograd, GradScaler
- Session 6: Tokenization, RNN, GRU, attention
- Session 7: Attention (self and cross), positional embeddings, transformer encoder/decoder, transfer learning using T5
- Session 8: GAN, conditional GAN, Wasserstein loss
- Session 9: Model initialization, TensorFlow/Keras, dataset, model creation, training, knowledge distillation
- Session 10: Time series forecasting with GRU in TensorFlow, quantization in PyTorch
- Session 11: Autoencoders, gradient checkpointing, optimizers

## Repository structure
- Slides: `s01-DLFrameworks.pdf`, `s02-DLFrameworks.pdf`, `s03-DLFrameworks.pdf` ... `s11-DLFrameworks.pdf`
- Notebooks: `s02.1-forward-backward-pass.ipynb`, `s02.2-mnist-lab.ipynb`

## Session 2
### Lab
- Lab 2.1 Forward/Backward Pass: https://www.kaggle.com/code/sakharam/forward-backward-pass
- Lab 2.2 MNIST Multi-Label Classification: https://www.kaggle.com/code/sakharam/mnist-lab

### Resources
- The Human Brain by Nancy Kanwisher: https://youtube.com/playlist?list=PLyGKBDfnk-iAQx4Kw9JeVqspbg77sfAK0&si=912O1zx96ZvmomJi
  - Refer recordings from 3.1 onwards on how images are processed by the eye and brain.
- Ultrascale Playbook: https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=high-level_overview
  - Tricks to use your compute efficiently.
- TensorFlow Playground: http://playground.tensorflow.org/
  - Explore how depth and width affect a neural network.

## Session 3
### Dataset
- Intel Scene for Image Classification: https://www.kaggle.com/datasets/puneet6060/intel-image-classification/

### Lab
- Lab 3.1 Gradient Clipping Example: https://www.kaggle.com/code/sakharam/gradient-clipping
- Lab 3.2 Transformations on Images: https://www.kaggle.com/code/sakharam/understanding-transforms-in-cnn
- Lab 3.3 Building Blocks of CNN: https://www.kaggle.com/code/sakharam/building-blocks-of-cnn
- Lab 3.4 CNN + saliency maps + class activation maps: https://www.kaggle.com/code/sakharam/cnn-explainability

## Session 4
### Dataset
- https://www.kaggle.com/datasets/jiweiliu/pennfudanped/

### Lab
- Lab 4 Object Detection: https://www.kaggle.com/code/sakharam/transfer-learning-pedestrian-detection

## Session 5
### Dataset
- https://www.kaggle.com/datasets/julinmaloof/the-oxfordiiit-pet-dataset

### Lab
- Lab 5.1 UNet: https://www.kaggle.com/code/sakharam/segmentation-u-net
- Lab 5.2 DeepLab: https://www.kaggle.com/code/sakharam/segmentation-deeplabv3

## Session 6
Note: Slide annotations were lost for this session.

### Lab
- Lab 6.1 News Classification: https://www.kaggle.com/code/sakharam/news-classification-gru
- Lab 6.2 Translation: https://www.kaggle.com/code/sakharam/fr-en-translation-gru-with-attention

## Session 7
### Dataset
- IMDB Reviews - Sentiment Analysis: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
- SAMSum - Summarization: https://www.kaggle.com/datasets/nileshmalode1/samsum-dataset-text-summarization

### Lab
- Lab 7.1 Sentiment Analysis: https://www.kaggle.com/code/sakharam/sentiment-imdb-transformers-encoder-from-scratch
- Lab 7.2 Summarization: https://www.kaggle.com/code/ameenhasan/text-summarization-using-t5
- Lab 7.3 (Extra) Sentiment Classification (SST2) using LoRA: https://www.kaggle.com/code/sakharam/gpt2-lora-peft-training-sentiment-analysis

### Resources
- Coding Transformers from Scratch: https://www.youtube.com/watch?v=ISNdQcPhsts

## Session 8
### Lab
- Lab 8.1 Conditional GAN: https://www.kaggle.com/code/sakharam/conditional-gan-fashion-mnist-generate-items

**Extra**
- Lab 8.2 Diffusion Models: https://www.kaggle.com/code/sakharam/diffusion-cifar
- Lab 8.3 Conditional Diffusion Models - Sketch to Pokemon: https://www.kaggle.com/code/sakharam/diffusion-model-sketch-to-pokemon

## Session 9
### Lab
- Lab 9.1 Fully Connected Network for FashionMNIST (PyTorch + TensorFlow)
  - PT: https://www.kaggle.com/code/sakharam/pytorch-fashion-mnist-multiclassclassification
  - TF: https://www.kaggle.com/code/sakharam/tensorflow-fashionmnist-multiclass-classificat
- Lab 9.2 (TF) Distillation using KL divergence: https://www.kaggle.com/code/sakharam/tensorflow-knowledge-distillation-fashion-mnist
- Lab 9.3 (TF) CNN - Cat or Dog: https://www.kaggle.com/code/sakharam/tensorflow-classification-cat-or-dog

### Resources
- Response-based Knowledge Distillation: https://arxiv.org/abs/1503.02531
- Intermediate Layers Knowledge Distillation: https://arxiv.org/abs/1412.6550
- Sequence-level Knowledge Distillation: https://aclanthology.org/D16-1139.pdf

## Session 10
### Lab
- Lab 10.1 Multivariate time series using GRU: https://www.kaggle.com/code/sakharam/multivariate-time-series-forecasting-with-gru
- Lab 10.2 Post-training quantization and quantization-aware training: https://www.kaggle.com/code/sakharam/pytorch-cnn-quantization-lab-static-qat

## Session 11
### Lab
- Lab 11.1 Convolutional Autoencoder (TensorFlow): https://www.kaggle.com/code/sakharam/convolutional-autoencoder-for-images-mnist
- Lab 11.2 Gradient Checkpointing (PyTorch): https://www.kaggle.com/code/sakharam/gradient-checkpointing-cnn-memory-optimization
