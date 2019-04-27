import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', default='dataset', help='the path of image data set')
parser.add_argument('--image_size', type=int, default=224, help='the height / width of the input image to network')
parser.add_argument('--batch_size', type=int, default=4, help='batch size')
parser.add_argument('--decay_epoch', type=int, default=10, help='learning rate decay start epoch num')
parser.add_argument('--lr', type=float, default=0.00004, help='learning rate')
parser.add_argument('--checkpoint_dir', default='checkpoints', help="path to saved models (to continue training)")
parser.add_argument('--sample_dir', default='samples', help='folder to output images and model checkpoints')
parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
