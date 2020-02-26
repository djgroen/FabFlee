from base.fab import *
from plugins.FabFlee.FabFlee import *

import chaospy as cp
import numpy as np
import easyvvuq as uq
import matplotlib.pyplot as plt
import pandas
from pandas import DataFrame
import sys
import os
from pprint import pprint
import json
# authors: Derek Groen, Diana Suleimenova, Wouter Edeling.


class custom_redirection(object):

    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # If you want the output to be visible immediately

    def flush(self):
        for f in self.files:
            f.flush()

tmp_dir = '/tmp/'
output_columns = ["Total error"]


def init_flee_campaign():

    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    flee_campaign = uq.Campaign(name='flee', work_dir=tmp_dir)

    params = {
        "awareness_level": {
            "type": "integer",
            "min": 0, "max": 2,
            "default": 1
        },
        "max_move_speed": {
            "type": "float",
            "min": 0.0, "max": 40000,
            "default": 200
        },
        "camp_move_chance": {
            "type": "float",
            "min": 0.0, "max": 1.0,
            "default": 0.001
        },
        "conflict_move_chance": {
            "type": "float",
            "min": 0.0,
            "max": 1.0,
            "default": 1.0
        },
        "default_move_chance": {
            "type": "float",
            "min": 0.0,
            "max": 1.0,
            "default": 0.3
        },
        "out_file": {
            "type": "string",
            "default": "out.csv"
        }
    }

    output_filename = params["out_file"]["default"]
    # Create an encoder, decoder and collation element for PCE test app
    encoder = uq.encoders.GenericEncoder(
        template_fname=get_plugin_path("FabFlee") +
        '/templates/simsetting.template',
        delimiter='$',
        target_filename='simsetting.csv'
    )
    decoder = uq.decoders.SimpleCSV(target_filename=output_filename,
                                    output_columns=output_columns,
                                    header=0)

    # Create a collation element for this campaign
    collater = uq.collate.AggregateSamples(average=True)

    # Add the Flee app
    flee_campaign.add_app(name="flee",
                          params=params,
                          encoder=encoder,
                          decoder=decoder,
                          collater=collater)

    # Create the sampler
    # Awareness_level is an integer, which does not work with SCSampler (only
    # with RandomSampler at the moment for which need to specify draw_samples
    # number)
    vary = {
        "max_move_speed": cp.Uniform(20, 500),
        #"camp_move_chance": cp.Uniform(0.0001, 1.0),
        #"conflict_move_chance": cp.Uniform(0.1, 1.0),
        #"default_move_chance": cp.Uniform(0.1, 1.0)
    }

    my_sampler = uq.sampling.SCSampler(vary=vary, polynomial_order=3)
    # Associate the sampler with the campaign
    flee_campaign.set_sampler(my_sampler)

    # Will draw all (of the finite set of samples)
    flee_campaign.draw_samples()

    flee_campaign.populate_runs_dir()

    # copy copy database file into flee_campaign directory
    campaign_db_file = flee_campaign.db_location.split(tmp_dir)[1]
    local("cp %s %s" % (os.path.join(tmp_dir, campaign_db_file),
                        os.path.join(flee_campaign.campaign_dir,
                                     campaign_db_file)
                        )
          )

    return flee_campaign


@task
def run_flee_easyvvuq(configs, simulation_periods=None, mode='parallel', ** args):
    """
    to run multiple conflict country, you can use :
        fab <remote_machine> run_flee_easyvvuq:'mali;ssudan_ccamp'
        fab <remote_machine> run_flee_easyvvuq:'mali;ssudan_ccamp', simulation_periods='10;60'

    simulation_periods parameter is optional
    Note : the number of elements in configs list should be same as simulation_periods
    if you want to run the default simulation_periods for some, just used empty string
    i.e,
    fab <remote_machine> run_flee_easyvvuq:'mali;ssudan_ccamp',simulation_periods='60;'
        simulation_periods(mali) = 60
        simulation_periods(ssudan_ccamp) = default value in run.py
    """
    configs = configs.split(';')
    if simulation_periods is None:
        simulation_periods = [-1 for _ in range(len(configs))]
    else:
        simulation_periods = [
            int(s) if s.isdigit() else -1 for s in simulation_periods.split(';')]

    if len(configs) != len(simulation_periods):
        print("number of conflict countries is NOT equal to simulation_periods")
        exit()

    if mode == 'serial':
        flee_script = 'flee'
    else:
        flee_script = 'pflee'
    # run once and used for all conflicts
    flee_campaign = init_flee_campaign()
    # flee_campaign.save_state(tmp_dir + "tmp.json")

    flee_campaign.save_state(os.path.join(
        flee_campaign.campaign_dir, "flee_easyvvuq_state.json"))

    for config, simulation_period in zip(configs, simulation_periods):
        campaign2ensemble(config, campaign_dir=flee_campaign.campaign_dir)

        if simulation_period == -1:
            flee_ensemble(config, script=flee_script, **args)
        else:
            flee_ensemble(config, simulation_period,
                          script=flee_script, **args)

        # copy EasyVVUQ folders into results
        with_config(config)
        name = template(env.job_name_template)
        local_dir = os.path.join(env.results_path, name)
        remote_dir = os.path.join(flee_campaign.campaign_dir)
        rsync_project(local_dir + '/', remote_dir)


def plot_grid(flee_analysis, keys):

    # find index of input keys in sampler.var
    if isinstance(keys, str):
        keys = [keys]
    key_indexs = []
    for key_value in keys:
        key_indexs.append(
            list(flee_analysis.sampler.vary.get_keys()).index(key_value))

    print(flee_analysis.sampler.vary.get_keys())
    print(key_indexs)
    print(len(key_indexs))

    if len(key_indexs) == 1:
        fig = plt.figure()
        ax = fig.add_subplot(111, xlabel='RUNs number', ylabel=keys[0])
        print(flee_analysis.xi_d[:, key_indexs[0]])
        ax.plot(flee_analysis.xi_d[:, key_indexs[0]], 'ro'
                )

    if len(key_indexs) == 2:
        fig = plt.figure()
        ax = fig.add_subplot(111, xlabel=keys[0], ylabel=keys[1])
        ax.plot(flee_analysis.xi_d[:, key_indexs[0]],
                flee_analysis.xi_d[:, key_indexs[1]],
                'ro'
                )
    elif len(key_indexs) == 3:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d', xlabel=keys[0],
                             ylabel=keys[1], zlabel=keys[2])
        ax.scatter(flee_analysis.xi_d[:, key_indexs[0]],
                   flee_analysis.xi_d[:, key_indexs[1]],
                   flee_analysis.xi_d[:, key_indexs[2]]
                   )
    else:
        print('Will only plot for N = 2 or N = 3.')

    plt.tight_layout()
    plt.show()


@task
def analyse_flee_easyvvuq(configs, ** args):
    # make sure you run fetch_results() command before this using this function

    configs = configs.split(';')
    for config in configs:
        with_config(config)
        # update_environment(args)

        state_file = None
        work_dir = os.path.join(
            env.local_results, template(env.job_name_template))
        # walk through all files in results_dir
        for root, dirs, files in os.walk(work_dir):
            dirs[:] = [d for d in dirs if d not in set(['RUNS'])]
            if ("flee_easyvvuq_state.json" in files):
                state_file = os.path.join(root, "flee_easyvvuq_state.json")
                state_folder = os.path.basename(root)
                break

        # read json file
        with open(state_file, "r") as infile:
            json_data = json.load(infile)
        # change database location file name in json file
        json_data['db_location'] = "sqlite:///" + \
            os.path.join(work_dir, state_folder, json_data[
                         'db_location'].split("/")[-1])        
        # save json file
        with open(state_file, "w") as outfile:
            json.dump(json_data, outfile, indent=4)

        # updating db file
        from sqlalchemy import create_engine
        engine = create_engine(json_data['db_location'])
        with engine.connect() as con:
            sql_cmd = "UPDATE app "
            sql_cmd += "SET input_encoder = JSON_SET(input_encoder,'$.state.template_fname','%s')" % (
                os.path.join(env.localplugins['FabFlee'], 'templates/simsetting.template'))
            result = con.execute(sql_cmd)
            result.close()

            sql_cmd = "UPDATE run "
            sql_cmd += "SET run_dir = '%s/'||run_name" % (
                os.path.join(work_dir, state_folder, 'runs'))
            result = con.execute(sql_cmd)
            result.close()

            sql_cmd = "UPDATE campaign_info "
            sql_cmd += "SET campaign_dir='%s' , runs_dir='%s'" % (
                os.path.join(work_dir, state_folder),
                os.path.join(work_dir, state_folder, 'runs')
            )
            result = con.execute(sql_cmd)
            result.close()
            
        flee_campaign = uq.Campaign(state_file=state_file, work_dir=work_dir)
    
        ensemble2campaign(
            "{}/{}".format(env.local_results, template(env.job_name_template)),
            campaign_dir=flee_campaign.campaign_dir
        )

        flee_campaign.collate()

        collation_result = flee_campaign.get_collation_result()
        print(collation_result)
        collation_result.to_csv(env.local_results + '/' +
                                template(env.job_name_template) +
                                '/collation_result.csv',
                                index=False
                                )

        # Post-processing analysis
        flee_analysis = uq.analysis.SCAnalysis(
            sampler=flee_campaign._active_sampler,
            qoi_cols=output_columns
        )

        flee_campaign.apply_analysis(flee_analysis)

        results = flee_campaign.get_last_analysis()

        mu = results['statistical_moments']['Total error']['mean']
        std = results['statistical_moments']['Total error']['std']

        flee_analysis_add_file = env.local_results + '/' + \
            template(env.job_name_template) + '/flee_analysis.txt'

        original = sys.stdout
        with open(flee_analysis_add_file, 'w') as file:
            sys.stdout = custom_redirection(sys.stdout, file)
            print(config)
            print(flee_analysis.samples)
            print('mean Total error = %f' % (mu))
            print('std Total error = %f' % (std))
            print('=================================================')
            print('Sobol indices:')
            # print all indices
            print(results['sobols']['Total error'])
            # print 1st order indices
            print(results['sobols_first']['Total error'])
            print('=================================================')

        sys.stdout = original

        # flee_analysis.plot_grid()
        plot_grid(flee_analysis, ['camp_move_chance',
                                  'conflict_move_chance',
                                  'default_move_chance'])

    # return results, flee_analysis
