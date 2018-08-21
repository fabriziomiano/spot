"""
This script filter a given dataset 

"""
from utils.misc import timestamp_filter
import argparse
import sys

args = sys.argv
parser = argparse.ArgumentParser(
    description="""Select sunset / sunrise images solely relying on their timestamp""")
parser.add_argument('-i', '--inDir', type=str, metavar='', required=True,
                    help='Specify the path of the datset to filter')
parser.add_argument('-o', '--outDir', type=str, metavar='', required=True,
                    help='Specify the destination path for the filtered dataset')
parser.add_argument('-tz', '--timezone', type=str, metavar='',
                    required=True, help='Specify the timezone')
parser.add_argument('-c', '--city', type=str, metavar='', required=True,
                    help='Specify the city name')
args = parser.parse_args()
PATH_IN = args.inDir
PATH_OUT = args.outDir
TIMEZONE = args.timezone
CITY_NAME = args.city
timestamp_filter(PATH_IN, PATH_OUT, TIMEZONE, CITY_NAME)
