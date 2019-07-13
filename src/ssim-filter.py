"""
This script filters a dataset of images 
in a given path using the SSIM score
against a reference image and copies
the images that pass a certain threshold
and tolerance to a selected output path 

"""

from .utils.misc import ssim_filter, load_config
import argparse
import sys


parser = argparse.ArgumentParser(
    description="""Select images solely relying on SSIM score against a reference image""")
parser.add_argument(
    '-c', '--config', type=str, metavar='', required=True,
    help='Specify the path of the configuration file')
args = parser.parse_args()
CONFIG_PATH = args.config
conf = load_config(CONFIG_PATH)
if len(conf.sections()) == 0:
    print("ERROR :: Not a valid configuration file. Exiting")
    sys.exit(0)
PATH_IN = conf.get('SSIM', 'PATH_IN')
PATH_OUT = conf.get('SSIM', 'PATH_OUT')
REF_IMG_PATH = conf.get('SSIM', 'REF_IMG_PATH')
THRESHOLD = float(conf.get('SSIM', 'THRESHOLD'))
ssim_filter(CONFIG_PATH, PATH_IN, PATH_OUT, REF_IMG_PATH, THRESHOLD)
