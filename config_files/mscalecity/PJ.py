'''
######################################################################
    Parse input arguments
######################################################################  
'''
import argparse
import sys
parser = argparse.ArgumentParser()
# Required parameters (mandatory)
parser.add_argument('--COUPLING_TYPE', required=True,
                    action="store", type=str,
                    choices=['file', 'muscle3'])
parser.add_argument('--WEATHER_COUPLING', required=True,
                    action="store", type=str)
parser.add_argument('--NUM_WORKERS', required=True,
                    action="store", type=int)
parser.add_argument('--WORKER_CORES', required=True,
                    action="store", type=int)
args, unknown = parser.parse_known_args()
print("args: {}".format(args), file=sys.stderr)

COUPLING_TYPE = args.COUPLING_TYPE
WEATHER_COUPLING = args.WEATHER_COUPLING
NUM_WORKERS = int(args.NUM_WORKERS)
WORKER_CORES = int(args.WORKER_CORES)
DATA_DIR = 'input_csv'
if WORKER_CORES > 1:
    PYTHON_CMD = "mpirun -n %d python3" % (WORKER_CORES)
else:
    PYTHON_CMD = "python3"

'''
######################################################################
    config PilotJob
######################################################################  
'''

from qcg.pilotjob.api.manager import Manager
from qcg.pilotjob.api.manager import LocalManager
from qcg.pilotjob.api.job import Jobs
m = LocalManager(cfg={'log_level': 'DEBUG'}, server_args=['--log', 'debug'])

# get available resources
print("\n\navailable resources:\n%s\n" % str(m.resources()))

# submit jobs and save their names in 'ids' list
jobs = Jobs()

print("Start Adding jobs . . .\n\n")

WORKER_INDEX = 0
for i in range(NUM_WORKERS):
    for SUBMODEL in ['macro', 'micro']:
        cmd = '%s run_couple.py --submodel %s --data_dir=%s --worker_index %d --coupling_type %s --num_workers %d --weather_coupling %s' % (
            PYTHON_CMD, SUBMODEL, DATA_DIR, WORKER_INDEX, COUPLING_TYPE, NUM_WORKERS, WEATHER_COUPLING)

        print("\tAdd job with cmd = %s" % (cmd))

        TaskID = 'TaskID%d_%s' % (WORKER_INDEX + 1, SUBMODEL)
        stderr = 'log_task/%s_${jname}__${uniq}.stderr' % (TaskID)
        stdout = 'log_task/%s_${jname}__${uniq}.stdout' % (TaskID)

        jobs.add(name=TaskID, exec='bash', args=['-c', cmd],
                 stdout=stdout, stderr=stderr,
                 numCores={'exact': WORKER_CORES}, model='default')
    WORKER_INDEX = WORKER_INDEX + 1


ids = m.submit(jobs)

# wait until submited jobs finish
m.wait4(ids)

# get detailed information about submited and finished jobs
print("jobs details:\n%s\n" % str(m.info(ids)))

m.finish()
m.kill_manager_process()
m.cleanup()
