import torch
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
# random seed
random_state = 7
torch.manual_seed(random_state)

import pathlib
# main path, save model, prediction, etc...
main_path = pathlib.Path('/repository/chaoyang_data/mnist/results')
# data path
# best model path
best_model_path = main_path.joinpath('best_model.pth')
# results csv file path
csv_header = ['truth', 'predict']
another_header = [f'score{i}' for i in range(10)]
csv_header.extend(another_header)
prediction = main_path.joinpath('prediction.csv')

# data params
# heigh, width, channel
h = 28
w = 28
c = 1
# number of seg class
n_class = 10

# train params
# batch size
batch_size = 128
# max epochs
max_epoch = 10
# the first epoch id, 1 or 0
start_epoch = 1
# learing rate
learing_rate = 1e-3
# milestones
milestones = [4, 6, 8, 9, 10]
# gamma
gamma = 0.1
