#!/bin/bash
source setup_env.sh
python3 -m hailo_apps.python.pipeline_apps.ddr.ddr --input usb
