#!/bin/bash
echo "====================================="
echo " Starting Teleoperation Bridge...    "
echo "====================================="
echo ""
echo "Connecting to Leader Arm..."
echo ""

source /home/iy/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
python /home/iy/Isaac/Robotics/robot_models/soarm_nbv/leader_teleop_bridge.py --leader-port /dev/ttyACM0 --leader-type so101_leader --leader-id teleop_leader_v1 --action-port 5556 --fps 60 --alpha 1.0
