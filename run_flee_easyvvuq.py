from base.fab import *
from plugins.FabFlee.FabFlee import *

import chaospy as cp
import numpy as np
import easyvvuq as uq
import matplotlib.pyplot as plt
import os
import time
import sys
# authors: Derek Groen, Diana Suleimenova, Wouter Edeling.

@task
def test_flee_easyvvuq(config, simulation_period, **args):
    tmp_dir = '/tmp/'

    my_campaign = uq.Campaign(name='flee', work_dir=tmp_dir)


    params = {
      "awareness_level": {"type": "integer", "min": 0, "max":2, "default":1},
      "max_move_speed": {"type": "float", "min": 0.0, "max":40000.0, "default":200.0},
      "camp_move_chance": {"type": "float", "min": 0.0, "max":1.0, "default":0.001},
      "conflict_move_chance": {"type": "float", "min": 0.0, "max":1.0, "default":1.0},
      "default_move_chance": {"type": "float", "min": 0.0, "max":1.0, "default":0.3},
      "out_file": {
            "type": "string",
            "default": "out.csv"}}

    output_filename = params["out_file"]["default"]
    output_columns = ["Total_error"]

    # Create an encoder, decoder and collation element for PCE test app
    encoder = uq.encoders.GenericEncoder(
        template_fname= get_plugin_path("FabFlee") + '/templates/simsetting.template',
        delimiter='$',
        target_filename='simsetting.csv')
    decoder = uq.decoders.SimpleCSV(target_filename=output_filename,
                                    output_columns=output_columns,
                                    header=0)

    # Create a collation element for this campaign
    collater = uq.collate.AggregateSamples(average=True)

    # Add the Flee app
    my_campaign.add_app(name="flee",
                        params=params,
                        encoder=encoder,
                        decoder=decoder,
                        collater=collater)
    
    # Create the sampler
    # Awareness_level is an integer, which does not work with SCSampler (only with RandomSampler at the moment for which need to specify draw_samples number)
    vary = {
        #"max_move_speed": cp.Uniform(20, 500),
        "camp_move_chance": cp.Uniform(0.0001, 1.0),
        "conflict_move_chance": cp.Uniform(0.1, 1.0),
        "default_move_chance": cp.Uniform(0.1, 1.0)
        }

    
    my_sampler = uq.sampling.SCSampler(vary=vary, polynomial_order=1)

    # Associate the sampler with the campaign
    my_campaign.set_sampler(my_sampler)

    # Will draw all (of the finite set of samples)
    my_campaign.draw_samples()

    my_campaign.populate_runs_dir()

    campaign2ensemble(config, campaign_dir=my_campaign.campaign_dir)
    flee_ensemble(config, simulation_period, **args)
    fetch_results()
    #TODO: do we need to explicitly point to local_results here?
    # Rewrite ensemble2campaign perhaps?

    ensemble2campaign("{}/{}".format(env.local_results,template(env.job_name_template)), campaign_dir=my_campaign.campaign_dir)


    # use code below when not running on localhost.
    #while job_stat(period="all"):
    #time.sleep(10)
 
    my_campaign.collate()
    
    print(my_campaign.get_collation_result())


    # Post-processing analysis
    flee_analysis = uq.analysis.SCAnalysis(sampler=my_sampler, qoi_cols=output_columns)

    my_campaign.apply_analysis(flee_analysis)

    results = my_campaign.get_last_analysis()
    #print(results)
    # Save and reload campaign
    #state_file = tmp_dir + "flee_state.json"
    #my_campaign.save_state(state_file)
    #new = uq.Campaign(state_file=state_file, work_dir=tmp_dir)
    #print(new)

    print(flee_analysis.samples)
    mu = results['statistical_moments']['Total_error']['mean']
    std = results['statistical_moments']['Total_error']['std']

    print(mu)
    print(std)

    print('=================================================')    
    print('Sobol indices:')
    #print all indices
    print(results['sobols']['Total_error'])
    #print 1st order indices
    print(results['sobols_first']['Total_error'])
    print('=================================================')    

    flee_analysis.plot_grid()

    return results, flee_analysis

