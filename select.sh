#!/bin/bash

function usage()
{
    echo ""
    echo -e "\tInstructions"
    echo ""
    echo -e "\tsource select.sh <path/to/config-file> <mode>"
    echo -e "\t<mode> can be either 'quick' or 'full'"
    echo ""
}

CONFIG=$1
MODE=$2
if [ -z $CONFIG ] || [ -z $MODE ] ; then
    echo
    echo "ERROR :: Options not specified. "
    echo "Please specify config-file path and mode "
    echo -e "\a"
    usage
elif [ $2 == "quick" ]; then
    echo "INFO :: Running SSIM selection only. Using file" $CONFIG
    python src/ssim-filter.py -c $CONFIG
elif [ $2 == "full" ]; then 
    echo "INFO :: Running timestamp selection first, then SSIM test"
    python src/timestamp-filter.py -c $CONFIG
    echo "INFO :: Running SSIM test now"
    python src/ssim-filter.py -c $CONFIG
else
    echo
    echo -e "ERROR :: Please choose either 'full' or 'quick'"
    echo -e "\a"
    usage
fi
