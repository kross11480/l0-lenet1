# Praktikum - Neural Network Fundamentals and Benchmarking

Understanding key concepts of machine learning and some basic familiarity with deep learning frameworks is necessary. In this praktikum, pytorch would be used to realize the first real world application of neural net **LeNet**.

---

## Learning Objectives:
1. Introduction to PyTorch
2. Understanding ML fundamentals with a real-world convolutional neural network from scratch
3. Benchmarking and optimization in neural networks

---
## Tasks

Open **`pytorch_playground`** notebook and work through the tasks below by completing the TODO markdown or code cells.

### Task 1: Tensor and Tensor Operations
Use one of the functions and visualize it with matplotlib.
* zeros()
* ones()
* randn()
* eye()
* others

### Task 2: Exploring the Dataset

MNIST is a dataset containing grayscale images of handwritten digits (0-9) with resolution of 28x28 pixels. 
* Find out the shape of an MNIST image. Visualize a sample image with its label using matplotlib.
* However, the initial paper had different number of training and test images and dimensions. The notebook is preprocessing the dataset and writes the preprocessed data in a file. Write in a markdown cell the characteristics of the dataset and in what format is the file?

Open **`lenet1`** notebook. Lenet-1 is the compact architecture introduced by LeCun et. al.

### Task 3: Loading the preprocessed dataset 
*Check the shape of the training and test data and labels loaded from data directory.

### Task 4: Defining Network parameters

you'll need to implement TODO section, with following specification for parameters
* H1 layer is a 2d Convolution layer with 12 filter with mask 5x5 and stride of 2
* H3 layer is fully connected layer (FC) where input shape is (1, 192) output shape is (1, 30)
* Final layer is also FC mapping to 10 digit classes
* activation function is tanh
* mac, act are the number of multiply-accumulate and activations, respectively.

### Task 5: calculating the memory and compute budget

Find out the following features of the model architecture
* number of parameters:
* number of macs: 
* number of activations:  

### Task 6: Implementing the forward pass of network architecture

you'll need to implement TODO section in forward pass, with following specification as in the paper
* H1 layer is a 2d Convolution layer with 12 filter with mask 5x5 and stride of 2. Padding is given.
* H3 layer is fully connected layer (FC) where input shape is (1, 192) output shape is (1, 30)
* Final layer is also FC mapping to 10 digit classes
* activation function is tanh

### Task 7: Update the loss function

* Update the loss function as in the paper (Mean Square Error)

### Task 8: Benchmarking Training and Inference

In 1989 the training of the CNN took few days.
* What is the training time of your implementation (number of epochs=23)
* Which layer of neural network needs most mac operations (arithmetic)
* How many bytes are transferred to/from memory by each layer considering the input output, and weights

---
## Getting started

---
## Evaluation Criteria
* report quality and analysis depth
* Team presentation (5–10 min)