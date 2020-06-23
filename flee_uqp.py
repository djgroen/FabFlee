from base.fab import *
import numpy as np
import os
import sys
import math
from pprint import pprint
import matplotlib.pyplot as plt
from plugins.FabFlee.FabFlee import *
import UQP.uqp as uqp

# Source file for UQP examples FabFlee.

def _flee_uqp(config_info, replicas, args):
  flee(config_info["config"], config_info["simulation_period"], replicas=replicas, args)
  fetch_results()
  return template(env.job_name_template)

def _flee_collate(output_directory):
  validate_flee_output(output_directory)


def test_uqp1(config,simulation_period,replicas=10, **args):
   uqp1_aleatoric(config, _flee_uqp, _flee_collate, replicas, args)


