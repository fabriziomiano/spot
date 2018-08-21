from utils.misc import ssim_filter
import argparse
import sys

args = sys.argv
parser = argparse.ArgumentParser(
    description="""Select images solely relying on SSIM score against a reference image""")
parser.add_argument('origin_path', type=str,
                    help='Specify the path of the datset to filter')
parser.add_argument('destination_path', type=str,
                    help='Specify the destination path for the filtered dataset')
parser.add_argument('ref_img_path', type=str,
                    help='Specify the path of the reference image')
parser.add_argument('threshold', type=float,
                    help='Specify the threshold for the SSIM test')
parser.add_argument('tolerance', type=float,
                    help='Specify the tolerance in decimal (e.g. 5%, 0.05)')
args = parser.parse_args()
PATH_IN = args.origin_path
PATH_OUT = args.destination_path
REF_IMG_PATH = args.ref_img_path
THRESHOLD = args.threshold
TOLERANCE = args.tolerance
ssim_filter(PATH_IN, PATH_OUT, REF_IMG_PATH, THRESHOLD, TOLERANCE)
