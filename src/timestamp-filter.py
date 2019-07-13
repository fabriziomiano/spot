"""
This script filters a dataset of images 
in a given path using timestamp information
from the file names. It copies the images that
pass the selection to a given output path. 
Please refer to the timestamp_filter() 
function in utils/misc.py for additional info.

"""
from .utils.misc import timestamp_filter, load_config
import argparse
import sys


parser = argparse.ArgumentParser(
    description="""Select sunset / sunrise images solely relying on their timestamp""")
parser.add_argument(
    '-c', '--config', type=str, metavar='', required=True,
    help='Specify the path of the configuration file')
args = parser.parse_args()
CONFIG_PATH = args.config
conf = load_config(CONFIG_PATH)
if len(conf.sections()) == 0:
    print("ERROR :: Not a valid configuration file. Exiting")
    sys.exit(0)
PATH_IN = conf.get('TIMESTAMP', 'PATH_IN')
PATH_OUT = conf.get('TIMESTAMP', 'PATH_OUT')
TIMEZONE = conf.get('TIMESTAMP', 'TIMEZONE')
CITY_NAME = conf.get('TIMESTAMP', 'CITY_NAME')
timestamp_filter(CONFIG_PATH, PATH_IN, PATH_OUT, TIMEZONE, CITY_NAME)
