"""
This script filters a dataset of images 
in a given path using the SSIM score
against a reference image and copies
the images that pass a certain threshold
and tolerance to a selected output path 

"""

from utils.misc import ssim_filter, load_config
import argparse
import ConfigParser
import sys

args = sys.argv
parser = argparse.ArgumentParser(
    description="""Select images solely relying on SSIM score against a reference image""")
parser.add_argument('-c', '--config', type=str, metavar='', required=True,
                    help='Specify the path of the configuration file')
args = parser.parse_args()
CONFIG_PATH = args.config
Config = load_config(CONFIG_PATH)
if len(Config.sections()) == 0:
    print "ERROR :: Not a valid configuration file. Exiting"
    sys.exit(0)
PATH_IN = Config.get('SSIM', 'PATH_IN')
PATH_OUT = Config.get('SSIM', 'PATH_OUT')
REF_IMG_PATH = Config.get('SSIM', 'REF_IMG_PATH')
THRESHOLD = float(Config.get('SSIM', 'THRESHOLD'))
ssim_filter(CONFIG_PATH, PATH_IN, PATH_OUT, REF_IMG_PATH, THRESHOLD)
