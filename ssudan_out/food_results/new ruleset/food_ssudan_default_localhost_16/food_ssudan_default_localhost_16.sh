# 
# Copyright (C) University College London, 2007-2014, all rights reserved.
# 
# This file is part of FabSim and is CONFIDENTIAL. You may not work 
# with, install, use, duplicate, modify, redistribute or share this
# file, or any part thereof, other than as allowed by any agreement
# specifically made by you with University College London.
# 
# no batch system


# 
# Copyright (C) University College London, 2007-2012, all rights reserved.
# 
# This file is part of HemeLB and is CONFIDENTIAL. You may not work 
# with, install, use, duplicate, modify, redistribute or share this
# file, or any part thereof, other than as allowed by any agreement
# specifically made by you with University College London.
# 
cd /home/christian/FabSim3/results/food_ssudan_default_localhost_16
echo Running...
export PYTHONPATH=~/Documents/Python/SSudan/ICCS/flee:$PYTHONPATH

/usr/bin/env > env.log
python3 run_food.py /home/christian/Documents/Python/SSudan/FabSim3/plugins/FabFlee/config_files/ssudan_default/input_csv /home/christian/Documents/Python/SSudan/FabSim3/plugins/FabFlee/config_files/ssudan_default/source_data 604 simsetting.csv > out.csv
