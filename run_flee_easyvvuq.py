from base.fab import *
from plugins.FabFlee.FabFlee import *

import chaospy as cp
import numpy as np
import easyvvuq as uq
import matplotlib.pyplot as plt
import sys
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

    flee_campaign = uq.Campaign(name='flee', work_dir=tmp_dir)

    params = {
        "awareness_level": {
            "type": "integer",
            "min": 0, "max": 2,
            "default": 1
        },
        "max_move_speed": {
            "type": "float",
            "min": 20.0, "max": 40000,
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
        "camp_move_chance": cp.Uniform(0.0001, 1.0),
        "conflict_move_chance": cp.Uniform(0.1, 1.0),
        "default_move_chance": cp.Uniform(0.1, 1.0)
    }

    my_sampler = uq.sampling.SCSampler(vary=vary, polynomial_order=3)
    # Associate the sampler with the campaign
    flee_campaign.set_sampler(my_sampler)

    # Will draw all (of the finite set of samples)
    flee_campaign.draw_samples()

    flee_campaign.populate_runs_dir()

    return flee_campaign


@task
def run_flee_easyvvuq(configs, simulation_periods=None, **args):
    """
    to run multiple conflict country, you can use :
        fab <remote_machine> run_flee_easyvvuq:'conflict1';conflict2'
        fab <remote_machine> run_flee_easyvvuq:'conflict1';conflict2', simulation_periods='10;60'

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

    # run once and used for all conflicts
    flee_campaign = init_flee_campaign()
    flee_campaign.save_state(tmp_dir + "tmp.json")

    campaign_db_name = flee_campaign.db_location.split(tmp_dir)[1]
    with open(tmp_dir + "tmp.json", "r") as infile:
        flee_campaign_json = json.load(infile)

    for config, simulation_period in zip(configs, simulation_periods):
        campaign2ensemble(config, campaign_dir=flee_campaign.campaign_dir)

        # copy database file
        campaign_new_db_name = campaign_db_name.replace(
            ".db", "_" + config + ".db")
        local("cp %s %s" % (tmp_dir + campaign_db_name,
                            tmp_dir + campaign_new_db_name
                            )
              )
        # change database location file name in json file
        flee_campaign_json['db_location'] = flee_campaign.db_location.replace(
            ".db", "_" + config + ".db")
        # save json file
        with open(
                find_config_file_path(config, ExceptWhenNotFound=False) +
                "/SWEEP/" + "flee_easyvvuq_state.json", "w"
        ) as outfile:
            json.dump(flee_campaign_json, outfile, indent=4)

        if simulation_period == -1:
            flee_ensemble(config, **args)
        else:
            flee_ensemble(config, simulation_period, **args)


@task
def test_flee_easyvvuq(configs, ** args):
    # make sure you run fetch_results() command before this using this function

    configs = configs.split(';')
    for config in configs:
        with_config(config)
        # update_environment(args)

        flee_campaign = uq.Campaign(
            state_file=find_config_file_path(config, ExceptWhenNotFound=False) +
            "/SWEEP/" + "flee_easyvvuq_state.json",
            work_dir=tmp_dir
        )

        ensemble2campaign(
            "{}/{}".format(env.local_results, template(env.job_name_template)),
            campaign_dir=flee_campaign.campaign_dir
        )

        
        flee_campaign.collate()
        continue
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

        flee_analysis.plot_grid()

    # return results, flee_analysis
