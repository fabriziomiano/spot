#!/bin/bash

function usage()
{
    echo ""
    echo -e "\tInstructions"
    echo ""
    echo -e "\tsource classify.sh <path/to/config-file>"
    echo ""
}

CONFIG=$1
if [ -z $CONFIG ]; then
    echo "ERROR :: Config file not specified"
    echo "Please specify the config-file path to use"
    echo -e "\a"
    usage
else
    echo "INFO :: Running classification"
    python src/label.py -c $CONFIG
fi

