from base.fab import *
import chaospy as cp
import numpy as np
import easyvvuq as uq
import matplotlib.pyplot as plt
import pandas
from pandas import DataFrame
import sys
import os
import math
from shutil import copyfile, rmtree
from pprint import pprint
import json
from plugins.FabFlee.FabFlee import *

# authors: Hamid Arabnejad, Diana Suleimenova, Wouter Edeling, Derek Groen


# load custom Campaign
from plugins.FabFlee.customEasyVVUQ import CustomCampaign
uq.Campaign = CustomCampaign

work_dir_SCSampler = os.path.join(
    os.path.dirname(__file__), 'flee_easyvvuq_SCSampler')
backup_dir = os.path.join(work_dir_SCSampler, 'backup')

output_columns = ["Total error"]


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


@task
def flee_init_SC(config, simulation_period=-1, mode='serial', ** args):
    '''
    ============================================================================

        fab <remote_machine> flee_init_SC:<conflict_name>,simulation_period=<number>

    example:

        fab eagle_vecma flee_init_SC:mali,simulation_period=100

    ============================================================================
    '''
    # delete work_dir_SCSampler is exists
    if os.path.exists(work_dir_SCSampler):
        rmtree(work_dir_SCSampler)
    os.mkdir(work_dir_SCSampler)

    # Set up a fresh campaign called "flee-SCSampler"
    campaign = uq.Campaign(name='flee-SCSampler',
                           work_dir=work_dir_SCSampler)

    # to make sure we are not overwriting the new simulation on previous ones
    job_label = campaign._campaign_dir

    # Define parameter space for the flee-SCSampler app
    params = json.load(open(os.path.join(get_plugin_path("FabFlee"),
                                         'templates',
                                         'params.json'
                                         )
                            )
                       )

    output_filename = params["out_file"]["default"]

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

    # Add the flee-SCSampler app
    campaign.add_app(name="flee-SCSampler",
                          params=params,
                          encoder=encoder,
                          decoder=decoder,
                          collater=collater)

    # loading user input sampler yaml file
    user_sampler_yaml_file = os.path.join(os.path.dirname(__file__),
                                          "sampler_input_parameters.yml")
    sampler_args = yaml.load(open(user_sampler_yaml_file),
                             Loader=yaml.SafeLoader
                             )

    # parameters to vary
    vary = {}
    for param in sampler_args['selected_parameters']:
        lower_value = sampler_args['parameters'][param]['cp_uniform'][0]
        upper_value = sampler_args['parameters'][param]['cp_uniform'][1]
        vary.update({param: cp.Uniform(lower_value, upper_value)})

    # create SCSampler
    sampler = uq.sampling.SCSampler(
        vary=vary,
        polynomial_order=sampler_args['polynomial_order'],
        quadrature_rule=sampler_args['quadrature_rule'],
        sparse=sampler_args['sparse'],
        growth=sampler_args['growth'],
        midpoint_level1=sampler_args['midpoint_level1'],
        dimension_adaptive=sampler_args['dimension_adaptive']
    )

    # Associate the sampler with the campaign
    campaign.set_sampler(sampler)

    # Will draw all (of the finite set of samples)
    campaign.draw_samples()

    run_ids = campaign.populate_runs_dir()

    # copy generated run folders to SWEEP directory in config folder
    # 1. clean config SWEEP dir
    # 2. copy all generated runs by easyvvuq to config SWEEP folder
    path_to_config = find_config_file_path(config)
    sweep_dir = path_to_config + "/SWEEP"

    if os.path.exists(sweep_dir):
        rmtree(sweep_dir)
    os.mkdir(sweep_dir)
    print('=' * 20)
    print("Copying easyvvuq runs to %s SWEEP folder ..." % (config))
    with hide('output', 'running', 'warnings'), settings(warn_only=True):
        local(
            "rsync -av -m -v \
            {}/  {} ".format(os.path.join(campaign.work_dir,
                                          campaign.campaign_dir,
                                          'SWEEP'), os.path.join(sweep_dir))
        )
    print("Done")
    print('=' * 20)

    if mode == 'serial':
        flee_script = 'flee'
    else:
        flee_script = 'pflee'

    if simulation_period == -1:
        flee_ensemble(config,
                      script=flee_script,
                      label=job_label,
                      **args)
    else:
        flee_ensemble(config,
                      simulation_period,
                      script=flee_script,
                      label=job_label,
                      **args)

    # save campaign and sampler state
    campaign.save_state(os.path.join(
        work_dir_SCSampler, "campaign_state.json"))
    sampler.save_state(os.path.join(
        work_dir_SCSampler, "campaign_sampler.pickle"))

    backup_campaign_files()


@task
def flee_analyse_SC(config, ** args):
    '''
    ============================================================================

        fab <remote_machine> flee_analyse_SC:<conflict_name>

    example:

        fab eagle_vecma flee_analyse_SC:mali

    ============================================================================
    '''

    load_campaign_files()

    # reload Campaign
    campaign = uq.Campaign(state_file=os.path.join(work_dir_SCSampler,
                                                   "campaign_state.json"),
                           work_dir=work_dir_SCSampler
                           )
    print('========================================================')
    print('Reloaded campaign', campaign._campaign_dir)
    print('========================================================')

    sampler = campaign.get_active_sampler()
    # sampler.load_state(os.path.join(work_dir_SCSampler, "campaign_sampler.pickle"))
    campaign.set_sampler(sampler)

    # fetch only the required folder from remote machine
    with_config(config)

    # fetch results from remote machine
    job_label = campaign._campaign_dir
    job_folder_name = template(env.job_name_template + "_{}".format(job_label))
    print("fetching results from remote machine ...")
    with hide('output', 'running', 'warnings'), settings(warn_only=True):
        fetch_results(regex=job_folder_name)
    print("Done\n")

    # copy only output folder into local campaign_dir :)
    src = os.path.join(env.local_results, job_folder_name, 'RUNS')
    des = os.path.join(work_dir_SCSampler, campaign._campaign_dir, 'SWEEP')

    print("Syncing output_dir ...")
    with hide('output', 'running', 'warnings'), settings(warn_only=True):
        local(
            "rsync -av -m -v \
            --include='/*/' \
            --include='out.csv'  \
            --exclude='*' \
            {}/  {} ".format(src, des)
        )
    print("Done\n")

    campaign.collate()

    # Return dataframe containing all collated results
    collation_result = campaign.get_collation_result()

    collation_result.to_csv(os.path.join(work_dir_SCSampler,
                                         'collation_result.csv'
                                         ),
                            index=False
                            )

    print(collation_result)

    # Post-processing analysis
    analysis = uq.analysis.SCAnalysis(
        sampler=campaign._active_sampler,
        qoi_cols=output_columns
    )
    campaign.apply_analysis(analysis)
    results = campaign.get_last_analysis()

    # --------------------------------------------------------------------------
    #                   Plotting
    # --------------------------------------------------------------------------
    analysis_log_file = os.path.join(work_dir_SCSampler, 'analysis_log.txt')

    for output_column in output_columns:
        fig = plt.figure()
        ax = fig.add_subplot(111,
                             xlabel="days", ylabel=output_column)
        mean = results.raw_data["statistical_moments"][output_column]["mean"]
        std = results.raw_data["statistical_moments"][output_column]["std"]
        ax.plot(mean)
        ax.plot(mean + std, '--r')
        ax.plot(mean - std, '--r')
        #ax.title.set_text('statistical_moments for {}'.format(output_column))

        plt.tight_layout()
        plt.savefig(os.path.join(
            work_dir_SCSampler, 'plot_statistical_moments_{}'.format(output_column)),
            dpi=400)

        # analysis.plot_grid()
        '''
        plot_grid(analysis, ['max_move_speed',
                             'max_walk_speed'],
                  os.path.join(work_dir_SCSampler, 'analysis_plot_grid')
                  )
        '''

        original = sys.stdout
        with open(analysis_log_file, 'w') as file:
            sys.stdout = custom_redirection(sys.stdout, file)
            print(config)
            print(analysis.samples)
            print('mean Total error = %f' % (mean))
            print('std Total error = %f' % (std))
            print('=================================================')
            print('Sobol indices:')
            # print all indices
            print(results.raw_data['sobols']['Total error'])
            # print 1st order indices
            print(results.raw_data['sobols_first']['Total error'])
            print('=================================================')

        sys.stdout = original


def backup_campaign_files():

    # delete backup folder
    if os.path.exists(backup_dir):
        rmtree(backup_dir)
    os.mkdir(backup_dir)
    with hide('output', 'running', 'warnings'), settings(warn_only=True):
        local(
            "rsync -av -m -v \
            --include='*.db' \
            --include='*.pickle' \
            --include='*.json' \
            --exclude='*' \
            {}/  {} ".format(work_dir_SCSampler, backup_dir)
        )


def load_campaign_files():

    with hide('output', 'running', 'warnings'), settings(warn_only=True):
        local(
            "rsync -av -m -v \
            --include='*.db' \
            --include='*.pickle' \
            --include='*.json' \
            --exclude='*' \
            {}/  {} ".format(backup_dir, work_dir_SCSampler)
        )


def plot_grid(flee_analysis, keys, file=None):

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
    if file == None:
        plt.show()
    else:
        plt.savefig(file, dpi=400)
