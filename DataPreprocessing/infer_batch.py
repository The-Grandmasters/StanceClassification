#!/usr/bin/env python3

"""
Kai-Li Cheng 10/4/2020
Based on infer_single, from https://github.com/anibali/margipose
project 
"""
import os
import argparse
import PIL.Image
import matplotlib.pylab as plt
import torch
from mpl_toolkits.mplot3d import Axes3D
from pose3d_utils.coords import ensure_cartesian

from pathlib import Path 
from cli import Subcommand
from skeleton import CanonicalSkeletonDesc
from data_specs import ImageSpecs
from models import load_model
from utils import seed_all, init_algorithms, plot_skeleton_on_axes3d

CPU = torch.device('cpu')


def parse_args(argv):
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(prog='margipose-infer',
                                     description='3D human pose inference',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=str, metavar='FILE', default=argparse.SUPPRESS,
                        required=True,
                        help='path to model file')
    parser.add_argument('--dir', type=str, metavar='', default=argparse.SUPPRESS,
                        required=True,
                        help='path to directory of MxM (square) images ')
    parser.add_argument('--out', type=str, metavar='', default=argparse.SUPPRESS,
                        required=True,
                       help='output directory')                

    args = parser.parse_args(argv[1:])

    return args


def main(argv, common_opts):
    args = parse_args(argv)
    seed_all(12345)
    init_algorithms(deterministic=True)
    torch.set_grad_enabled(False)

    # Load the model 
    device = common_opts['device']
    model = load_model(args.model).to(device).eval()
    input_specs: ImageSpecs = model.data_specs.input_specs


    pathlist = Path(args.dir).glob('**/*.jpg')
    for path in pathlist:
        imgPath = str(path) 
        image: PIL.Image.Image = PIL.Image.open(imgPath, 'r')
        image.thumbnail((input_specs.width, input_specs.height))
        inp = input_specs.convert(image).to(device, torch.float32)

        output = model(inp[None, ...])[0]
        out = imgPath.split("/")[-1].split(".")[0] + ".pt"
        print(output, " ", os.path.join(args.out, out))
        torch.save(output, os.path.abspath(os.path.join(args.out, out)))
        norm_skel3d = ensure_cartesian(output.to(CPU, torch.float64), d=3)

        fig = plt.figure(figsize=(16, 8))
        ax1 = fig.add_subplot(1, 2, 1)
        ax2: Axes3D = fig.add_subplot(1, 2, 2, projection='3d')

      
        #ax1.imshow(input_specs.unconvert(inp.to(CPU)))
        #plot_skeleton_on_axes3d(norm_skel3d, CanonicalSkeletonDesc, ax2, invert=True)
        #plt.show()

        


Infer_Subcommand = Subcommand(name='infer_batch', func=main, help='infer 3D pose for a directory of "square" images')

if __name__ == '__main__':
    Infer_Subcommand.run()

