
from fabsim.base.decorators import ptask
from fabsim.base.environment_manager import env
from FabFlee import *


@ptask
def flee_get_perf(results_dir):
    print("{}/{}".format(env.local_results, results_dir))
    my_file = open(
        "{}/{}/perf.log".format(env.local_results, results_dir), 'r')
    print(my_file.read())
