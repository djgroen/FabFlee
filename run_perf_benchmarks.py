from base.fab import *
from plugins.FabFlee.FabFlee import *

@task
def flee_get_perf(results_dir):
    print("{}/{}".format(env.local_results,results_dir))
    my_file = open("{}/{}/perf.log".format(env.local_results,results_dir), 'r')
    print(my_file.read())
