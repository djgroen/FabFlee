# FabFlee
FabFlee is a plugin for automated [Flee](https://github.com/djgroen/flee)-based simulations. It provides an environment to construct, modify and execute simulations as a single run or ensemble runs. FabFlee aims to predict the distribution of incoming refugees across destination camps under a range of different policy situations.

## Installation
Simply type `fabsim localhost install_plugin:FabFlee` anywhere inside your FabSim3 to install FabFlee directory.

## Explanation of files
* FabFlee.py - main file containing various command implementations.
* templates/flee - template file for running the Flee command on the local and remote machine.
* conflict_data/<conflict_name> - directory containing base conflict scenarios, such as ABC - an example conflict, CAR - Central African Republic and SSudan - South Sudan. Each conflict directory consists of source data and input files.

## Working with FabFlee 

The main tutorial for the VECMAtk release can be found in https://github.com/djgroen/FabFlee/blob/master/doc/FabFlee.md.
