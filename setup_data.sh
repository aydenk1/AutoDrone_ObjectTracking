#!/bin/bash

set -e

result=$(wget -q --show-progress "https://purdue0-my.sharepoint.com/:u:/g/personal/akocher_purdue_edu/EZnfZIRgkxtHjhnhNNc6IvABBrfuz4xDwmzcoezCqbwSNA?e=tVlHuZ&download=1" -O rc_car_PASCAL_VOC.zip)
echo "$result"
unzip rc_car_PASCAL_VOC.zip
rm rc_car_PASCAL_VOC.zip

