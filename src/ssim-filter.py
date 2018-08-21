from utils.misc import ssim_filter
import argparse
import sys

args = sys.argv
parser = argparse.ArgumentParser(
    description="""Select images solely relying on SSIM score against a reference image""")
parser.add_argument('-i', '--inDir', type=str, metavar='',
                    help='Specify the path of the datset to filter')
parser.add_argument('-o', '--outDir', type=str, metavar='',
                    help='Specify the destination path for the filtered dataset')
parser.add_argument('-r','--ref-img', type=str, metavar='',
                    help='Specify the path of the reference image')
parser.add_argument('-th','--threshold', type=float, metavar='',
                    help='Specify the threshold for the SSIM test')
parser.add_argument('-to','--tolerance', type=float, metavar='',
                    help='Specify the tolerance in decimal (e.g. 0.05)')
args = parser.parse_args()
PATH_IN = args.inDir
PATH_OUT = args.destination_path
REF_IMG_PATH = args.ref_img_path
THRESHOLD = args.threshold
TOLERANCE = args.tolerance
ssim_filter(PATH_IN, PATH_OUT, REF_IMG_PATH, THRESHOLD, TOLERANCE)
