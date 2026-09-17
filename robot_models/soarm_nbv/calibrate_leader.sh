#!/bin/bash
echo "====================================="
echo " Starting Leader Arm Calibration...  "
echo "====================================="
echo ""
echo "This will use the official LeRobot calibration tool."
echo "Please follow the interactive instructions on the screen."
echo ""

source /home/iy/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
python /home/iy/Isaac/lerobot/src/lerobot/scripts/lerobot_calibrate.py --teleop.type=so101_leader --teleop.port=/dev/ttyACM0 --teleop.id=teleop_leader_v1
