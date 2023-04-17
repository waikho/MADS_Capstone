#!/bin/bash
source /home/ubuntu/capstone/bin/activate
rm -f /home/ubuntu/capstone/app/nohup.out
(cd /home/ubuntu/capstone/app && nohup python3 main.py > nohup.out &)