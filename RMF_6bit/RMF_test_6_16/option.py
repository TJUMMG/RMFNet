import argparse


parser = argparse.ArgumentParser(description='RMF')

# Hardware specifications
parser.add_argument('--n_threads', type=int, default=4,
                    help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')
parser.add_argument('--gpu_id', type=str, default='0',
                    help='gpu_id')

# Data specifications
parser.add_argument('--flip_aug', default='True',
                    help='flip augmentation')
parser.add_argument('--scale_aug', action='store_true',
                    help='flip augmentation')
parser.add_argument('--rot_aug', default='True',
                    help='rot augmentation')
parser.add_argument('--floor_loss', action='store_true',
                    help='floor loss')
parser.add_argument('--bn', default=False,
                    help='Batch Normalization')
                    
parser.add_argument('--dir_data', type=str, default='./Dataset/',
                    help='dataset directory')
parser.add_argument('--data_train', type=str, default='TrainData',
                    help='train dataset name')
parser.add_argument('--data_test', type=str, default='UST-HK',
                    help='test dataset name')
parser.add_argument('--n_train', type=int, default=80200,
                    help='number of training set')
parser.add_argument('--n_val', type=int, default=2310,
                    help='number of validation set')
parser.add_argument('--offset_val', type=int, default=80200,
                    help='validation index offest')
parser.add_argument('--ext', type=str, default='img',
                    help='dataset file extension')
parser.add_argument('--scale', type=int, default=1,
                    help='super resolution scale')
parser.add_argument('--patch_size', type=int, default=256,
                    help='output patch size')
parser.add_argument('--rgb_range', type=int, default=65535,
                    help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--in_channels', type=int, default=3,
                    help='number of in channels for isp')
parser.add_argument('--chop', action='store_true',
                    help='enable memory-efficient forward')
parser.add_argument('--testbin', default=False,
                    help='benchmark useing bin data')
# Model specifications
parser.add_argument('--model', default='RMF',
                    help='model name')

parser.add_argument('--act', type=str, default='lrelu',
                    help='activation function')
parser.add_argument('--pre_train', type=str, default='.',
                    help='pre-trained model directory')
parser.add_argument('--extend', type=str, default='.',
                    help='pre-trained model directory')
parser.add_argument('--b_blocks', type=int, default=2,
                    help='number of residual blocks')
parser.add_argument('--a_blocks', type=int, default=2,
                    help='number of residual blocks')


parser.add_argument('--n_feats', type=int, default=32,
                    help='number of feature maps')
parser.add_argument('--res_scale', type=float, default=1,
                    help='residual scaling')
parser.add_argument('--precision', type=str, default='single',
                    choices=('single', 'half'),
                    help='FP precision for test (single | half)')


# Training specifications
parser.add_argument('--reset', action='store_true',
                    help='reset the training')
parser.add_argument('--test_every', type=int, default=20050,
                    help='do test per every N batches')
parser.add_argument('--epochs', type=int, default=70,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=4,
                    help='input batch size for training')
parser.add_argument('--split_batch', type=int, default=1,
                    help='split the batch into smaller chunks')
parser.add_argument('--test_only', action='store_true',
                    help='set this option to test the model')


# Optimization specifications
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--lr_decay', type=str, default='35',
                    help='learning rate decay per N epochs')
parser.add_argument('--decay_type', type=str, default='step',
                    help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999),
                    help='ADAM beta')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')

# Loss specifications
parser.add_argument('--loss', type=str, default='1*lossK3+1*lossK2+1*lossK1+1*L1',
                    help='loss function configuration')
parser.add_argument('--skip_threshold', type=float, default='1e6',
                    help='skipping batch that has large error')

# Log specifications
parser.add_argument('--save', type=str, default='test',
                    help='file name to save')
parser.add_argument('--load', type=str, default='test',
                    help='file name to load')
parser.add_argument('--resume', type=int, default=-1,
                    help='resume from specific checkpoint')
parser.add_argument('--print_model', action='store_true',
                    help='print model')
parser.add_argument('--save_models', action='store_true',
                    help='save all intermediate models')
parser.add_argument('--print_every', type=int, default=2005,
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_results', action='store_true',
                    help='save output results')
parser.add_argument('--save_path', default='Results',
                    help='save output results')
parser.add_argument('--save_gt', action='store_true',
                    help='save low-resolution and high-resolution images together')

args = parser.parse_args()



if args.epochs == 0:
    args.epochs = 1e8

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False

