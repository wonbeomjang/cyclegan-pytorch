import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', default='dataset', help='the path of image data set')
parser.add_argument('--image_size', type=int, default=224, help='the height / width of the input image to network')
parser.add_argument('--batch_size', type=int, default=4, help='batch size')
parser.add_argument('--decay_epoch_1', type=float, default=0.5, help='adam: learning rate decay start epoch num')
parser.add_argument('--decay_epoch_2', type=float, default=0.999, help='adam: learning rate decay start epoch num')
parser.add_argument('--decay_epoch', type=int, default=100, help='learning rate decay start epoch num')
parser.add_argument('--num_epoch', type=int, default=200, help='learning rate decay start epoch num')
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--checkpoint_dir', default='checkpoints', help="path to saved models (to continue training)")
parser.add_argument('--sample_dir', default='samples', help='folder to output images and model checkpoints')
parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
parser.add_argument('--num_residual_blocks', type=int, default=4, help='number of data loading workers')
parser.add_argument('--lambda_cycle', type=float, default=5.0, help='number of data loading workers')
parser.add_argument('--lambda_identity', type=float, default=5.0, help='number of data loading workers')
parser.add_argument('--from_style', default="photo", help='')
parser.add_argument('--to_style', default="monet", help='')
parser.add_argument('--sample_epoch', type=int, default=23, help='')


def get_config():
    return parser.parse_args()

