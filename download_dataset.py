import os
import argparse
import shutil
from glob import glob
from utils.utils import download_url, unzip_zip_file

parser = argparse.ArgumentParser()
parser.add_argument('--style', type=str, default='monet2photo', help='the path of image data set')
args = parser.parse_args()

monet2photo_url = 'https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/monet2photo.zip'
apple2orange_url = 'https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/apple2orange.zip'

from_style = args.style.split('2')[0]
to_style = args.style.split('2')[1]

if not os.path.exists('temp'):
    os.makedirs('temp')
if not os.path.exists('dataset'):
    os.makedirs('dataset')
if not os.path.exists(os.path.join('dataset', from_style)):
    os.makedirs(os.path.join('dataset', from_style))
if not os.path.exists(os.path.join('dataset', to_style)):
    os.makedirs(os.path.join('dataset', to_style))


download_url(monet2photo_url, os.path.join('temp', f'{args.style}.zip'))
unzip_zip_file(os.path.join('temp', f'{args.style}.zip'), 'temp')

images_A = os.listdir(os.path.join('temp', args.style, 'trainA'))
images_B = os.listdir(os.path.join('temp', args.style, 'trainB'))

print('[*] Move image')
for image_name in images_A:
    shutil.move(os.path.join('temp', args.style, 'trainA', image_name), os.path.join('dataset', from_style, image_name))
for image_name in images_A:
    shutil.move(os.path.join('temp', args.style, 'trainB', image_name), os.path.join('dataset', to_style, image_name))



