"""
	Authors: Aiden Nibali, Jaggernaut
	Git Repository: https://github.com/anibali/margipose
"""
import argparse
from pathlib import Path
from PIL import Image, ImageOps
import sys
import os

def parse_args(argv):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-dir', type=str, metavar='', default=argparse.SUPPRESS,
                        required=True,
                        help='path to directory of images ')              

    args = parser.parse_args(argv[1:])
    return args

# Usage: python mirrorDir.py -dir (directory containing images to flip)
def main(argv):
    args = parse_args(argv)

    pathlist = Path(args.dir).glob('**/*.jpg')
    for path in pathlist:
        imgPath = str(path) 
        image: Image.Image = Image.open(imgPath, 'r')
        out = imgPath.split("/")[-1].split(".")[0] + "_flip.jpg"
        im_mirror = ImageOps.mirror(image)
        im_mirror.save(os.path.abspath(os.path.join(args.dir, out )), quality=95)

if __name__ == '__main__':
    main(sys.argv[0:])
