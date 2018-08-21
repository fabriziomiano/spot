from utils.misc import timestamp_filter
import argparse
import sys

args = sys.argv
parser = argparse.ArgumentParser(
    description="""Select sunset / sunrise images solely relying on their timestamp""")
parser.add_argument('origin_path', type=str,
                    help='Specify the path of the datset to filter')
parser.add_argument('destination_path', type=str,
                    help='Specify the destination path for the filtered dataset')
parser.add_argument('timezone', type=str,
                    help='Specify the timezone')
parser.add_argument('city_name', type=str,
                    help='Specify the city name')
args = parser.parse_args()
PATH_IN = args.origin_path
PATH_OUT = args.destination_path
TIMEZONE = args.timezone
CITY_NAME = args.city_name
timestamp_filter(PATH_IN, PATH_OUT, TIMEZONE, CITY_NAME)
