import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import config
from resnet import ResNet, BasicBlock
from dataset import Cifar_loader
from torch.utils.tensorboard import SummaryWriter
from time import process_time
from train import network_train, network_eval
from torchsummary import summary
from sklearn.model_selection import ParameterGrid

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-batch', type=int, default=64, help='dataloader batch size')
    parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('-w', type=int, default=0, help='number of workers for dataloader')
    parser.add_argument('-l2', type=float, default=0, help='l2 regularization')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Building ResNet model with batch={args.batch}, lr={args.lr}, l2={args.l2}')
    net = ResNet(BasicBlock, config.layers).to(device)
    summary(net, (3, 32, 32))

    print('Preparing CIFAR100 dataset...')
    path = config.data_path
    train_data = Cifar_loader(path,
                              batch_size=args.batch,
                              shuffle=True,
                              train=True,
                              num_workers=args.w)
    test_data = Cifar_loader(path,
                             batch_size=args.batch,
                             shuffle=True,
                             num_workers=args.w)

    loss_function = nn.CrossEntropyLoss()

    optimizer = optim.Adam(net.parameters(),
                           lr=args.lr,
                           weight_decay=args.l2)

    train_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', verbose=True)

    # creating the path to store the model
    save_checkpoint = os.path.join(config.save_path, config.model)
    if not os.path.exists(save_checkpoint):
        os.makedirs(save_checkpoint)

    # creating the path for tensorboard
    if not os.path.exists(config.log_dir):
        os.mkdir(config.log_dir)

    comment = f' batch_size={args.batch}, lr={args.lr}, l2={args.l2}'
    tb = SummaryWriter(log_dir=os.path.join(config.log_dir, f'{config.model}-{comment}'))

    print('Starting training...')
    for epoch in range(config.epoch):
        print(f'Start of epoch {epoch + 1}')
        start = process_time()

        train_loss, train_acc = network_train(net, train_data, optimizer, loss_function, device)
        test_loss, test_acc = network_eval(net, test_data, loss_function, device)

        train_scheduler.step(test_acc)

        print(f'Epoch took : {process_time() - start} seconds.')
        print('-----')

        tb.add_scalar('Train Loss', train_loss, epoch)
        tb.add_scalar('Train Accuracy', train_acc, epoch)
        tb.add_scalar('Test Loss', test_loss, epoch)
        tb.add_scalar('Test Accuracy', test_acc, epoch)
        for name, weight in net.named_parameters():
            tb.add_histogram(name, weight, epoch)
            tb.add_histogram(f'{name}.grad', weight.grad, epoch)
        tb.close()

        if epoch % 10 == 0:
            torch.save(net.state_dict(), os.path.join(save_checkpoint, f'{args.batch}-{args.lr}-{args.l2}.pth'))

    torch.save(net.state_dict(), os.path.join(save_checkpoint, f'{args.batch}-{args.lr}-{args.l2}.pth'))
