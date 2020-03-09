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
cd /home/cspgdds/FabSim3/results/ssudan_ccamp_localhost_16
echo Running...
export PYTHONPATH=~/Codes/flee:$PYTHONPATH
rm -rf results

cp -r /home/cspgdds/FabSim3/config_files/ssudan_ccamp/* .
/usr/bin/env > env.log
python3 run.py /home/cspgdds/Codes/FabSim3/plugins/FabFlee/config_files/ssudan_ccamp/input_csv /home/cspgdds/Codes/FabSim3/plugins/FabFlee/config_files/ssudan_ccamp/source_data 604 > out.csv
