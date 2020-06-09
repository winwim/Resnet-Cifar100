# Resnet-Cifar100
Training Resnet on Cifar100

This is a resnet practice on cifar100 dataset.


## Usage
Download Cifar100 dataset from their website https://www.cs.toronto.edu/~kriz/cifar.html and unzip it.
In config.py change the number of layers if you want(by default resnet18), set number of epochs, give the path to the unzipped folder with data. Also choose the path were the weights of model will be saved. Directory for tensorboard can also be given.

Run main.py for training.
The models will be trained on the gpu if it is available.
Control the training process by choosing hyperparameters: learning rate, batch size, l2 penalty. Also you can specify number of workers for DataLoader instance.

For example.
``` 
$ python main.py -batch=128 -lr=0.1 -w=2 -l2=2e-3

```
