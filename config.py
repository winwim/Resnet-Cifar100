# number of blocks for net, resnet18 by default
layers = [1, 1, 1, 1]
model = f'resnet{sum([2 * i for i in layers]) + 2}'

# path to save model weights
save_path = 'save_point'

# training epochs
epoch = 200

# tensorboard log_dir path
log_dir = 'runs'

# path to dataset
data_path = './data'


