from base.fab import *
import numpy as np
import easyvvuq as uq
import chaospy as cp
import os
import sys
import pytest
import math
from pprint import pprint
import subprocess
import json
from shutil import copyfile, rmtree
from sqlalchemy import create_engine
import re
import pandas as pd
import matplotlib.pyplot as plt
from plugins.FabFlee.FabFlee import *

try:
    import glob
except ImportError:
    raise ImportError('python glob module NOT found !! ')

# authors: Hamid Arabnejad, Diana Suleimenova, Wouter Edeling, Derek Groen


# load custom Campaign
from plugins.FabFlee.customEasyVVUQ import CustomCampaign, CustomSCAnalysis
uq.Campaign = CustomCampaign
uq.analysis.SCAnalysis = CustomSCAnalysis

'''
==================================================================
order to be executed
    fab eagle_vecma flee_adapt_init:mali,simulation_period=100
    fab eagle_vecma flee_adapt_analyse:mali
    loop
        fab eagle_vecma flee_adapt_look_ahead:mali,simulation_period=100
        fab eagle_vecma flee_adapt_dimension:mali
==================================================================        
'''
work_dir_adapt = os.path.join(os.path.dirname(__file__),
                              'flee_easyvvuq_adaptive')
backup_dir = os.path.join(work_dir_adapt, 'backup')


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
def flee_adapt_dimension(config, ** args):
    '''
    fab eagle_vecma flee_adapt_dimension:GB_suppress
    '''
    '''
    ============================================================================

        fab <remote_machine> flee_adapt_dimension:<conflict_name>

    example:

        fab eagle_vecma flee_adapt_dimension:mali

    ============================================================================
    '''
    load_campaign_files()

    # reload Campaign, sampler, analysis
    campaign = uq.Campaign(state_file=os.path.join(work_dir_adapt,
                                                   "campaign_state.json"),
                           work_dir=work_dir_adapt
                           )
    print('========================================================')
    print('Reloaded campaign', campaign._campaign_dir)
    print('========================================================')
    sampler = campaign.get_active_sampler()
    sampler.load_state(os.path.join(work_dir_adapt, "campaign_sampler.pickle"))
    campaign.set_sampler(sampler)
    analysis = uq.analysis.SCAnalysis(sampler=sampler, qoi_cols=output_columns)
    analysis.load_state(os.path.join(
        work_dir_adapt, "campaign_analysis.pickle"))

    # fetch only the required folder from remote machine
    with_config(config)
    # env.job_name_template += "_{}".format(job_label)
    # fetch results from remote machine
    job_label = campaign._campaign_dir
    job_folder_name = template(env.job_name_template + "_{}".format(job_label))

    print("fetching results from remote machine ...")
    # with hide('output', 'running', 'warnings'), settings(warn_only=True):
    fetch_results(regex=job_folder_name)
    print("Done\n")

    # copy only output folder into local campaign_dir :)
    src = os.path.join(env.local_results, job_folder_name, 'RUNS')
    des = os.path.join(work_dir_adapt, campaign._campaign_dir, 'SWEEP')

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

    # compute the error at all admissible points, select direction with
    # highest error and add that direction to the grid
    data_frame = campaign.get_collation_result()
    # for output_column in output_columns[0]:
    for output_column in [output_columns[0]]:
        analysis.adapt_dimension(output_column, data_frame)

    # save everything
    campaign.save_state(os.path.join(work_dir_adapt, "campaign_state.json"))
    sampler.save_state(os.path.join(work_dir_adapt, "campaign_sampler.pickle"))
    analysis.save_state(os.path.join(
        work_dir_adapt, "campaign_analysis.pickle"))

    # apply analysis
    campaign.apply_analysis(analysis)
    results = campaign.get_last_analysis()

    # for output_column in output_columns:
    for output_column in [output_columns[0]]:
        #########################
        # plot mean +/- std dev #
        #########################
        fig = plt.figure()
        ax = fig.add_subplot(111, xlabel="days", ylabel=output_column)
        mean = results["statistical_moments"][output_column]["mean"]
        std = results["statistical_moments"][output_column]["std"]
        ax.plot(mean)
        ax.plot(mean + std, '--r')
        ax.plot(mean - std, '--r')
        plt.tight_layout()

        plt.savefig(os.path.join(work_dir_adapt,
                                 'plot_mean_std_%d[%s]' %
                                 (sampler.number_of_adaptations,
                                  output_column)
                                 ),
                    dpi=400)

        #################################
        # Plot some convergence metrics #
        #################################
        # plot max quad order per dimension. Gives an idea of which
        # variables are important
        analysis.adaptation_histogram(
            os.path.join(work_dir_adapt,
                         'plot_adaptation_histogram_%d[%s]'
                         % (sampler.number_of_adaptations, output_column)
                         )
        )

        analysis.plot_stat_convergence(
            os.path.join(work_dir_adapt,
                         'plot_stat_convergence%d[%s]'
                         % (sampler.number_of_adaptations, output_column)
                         )
        )

        surplus_errors = analysis.get_adaptation_errors()

        fig = plt.figure()
        ax = fig.add_subplot(111, xlabel='refinement step',
                             ylabel='max surplus error')
        ax.plot(range(1, len(surplus_errors) + 1), surplus_errors, '-b*')
        plt.tight_layout()

        plt.savefig(os.path.join(work_dir_adapt,
                                 'max_surplus_error_%d[%s]' %
                                 (sampler.number_of_adaptations,
                                  output_column)
                                 ),
                    dpi=400)

        #####################################
        # Plot the random surrogate samples #
        #####################################

        fig = plt.figure(figsize=[12, 4])
        ax = fig.add_subplot(131, xlabel='days', ylabel=output_column,
                             title='Surrogate samples')
        ax.plot(analysis.get_sample_array(
            output_column).T, 'ro', alpha=0.5)

        # generate n_mc samples from the input distributions
        n_mc = 20
        xi_mc = np.zeros([n_mc, sampler.xi_d.shape[1]])
        idx = 0
        for dist in sampler.vary.get_values():
            xi_mc[:, idx] = dist.sample(n_mc)
            idx += 1
        xi_mc = sampler.xi_d
        n_mc = sampler.xi_d.shape[0]

        # evaluate the surrogate at these values
        print('Evaluating surrogate model', n_mc, 'times')
        for i in range(n_mc):
            ax.plot(analysis.surrogate(output_column, xi_mc[i]), 'g')
        print('done')

        plt.savefig(os.path.join(work_dir_adapt,
                                 'Surrogate_samples_%d[%s]' %
                                 (sampler.number_of_adaptations,
                                  output_column)
                                 ),
                    dpi=400)

        ##################################
        # Plot first-order Sobol indices #
        ##################################

        ax = fig.add_subplot(122, title=r'First-order Sobols indices',
                             xlabel="days", ylabel=output_column)
        sobols_first = results["sobols_first"][output_column]
        for param in sobols_first.keys():
            ax.plot(sobols_first[param], label=param)
        leg = ax.legend(loc=0, fontsize=8)
        leg.set_draggable(True)
        plt.tight_layout()

        plt.savefig(os.path.join(work_dir_adapt, 'plot_first_order_Sobol_indices_%d' %
                                 (sampler.number_of_adaptations)), dpi=400)

        ##################################
        # analysis.mean_history #
        ##################################
        plt.clf()
        ax = fig.add_subplot(111, xlabel='plot_analysis.mean_history.T')

        ax.plot(np.array(analysis.mean_history).T)

        plt.tight_layout()
        plt.savefig(os.path.join(work_dir_adapt, 'plot_analysis_mean_history_%d' %
                                 (sampler.number_of_adaptations)), dpi=400)

        # pprint(analysis.std_history)

    backup_campaign_files()


@task
def flee_adapt_look_ahead(config, simulation_period, mode='parallel', ** args):
    '''
    ============================================================================

        fab <remote_machine> flee_adapt_look_ahead:<conflict_name>,simulation_period=<number>

    example:

        fab eagle_vecma flee_adapt_look_ahead:mali,simulation_period=100

    ============================================================================
    '''

    load_campaign_files()

    # reload Campaign, sampler, analysis
    campaign = uq.Campaign(state_file=os.path.join(work_dir_adapt,
                                                   "campaign_state.json"),
                           work_dir=work_dir_adapt
                           )
    print('========================================================')
    print('Reloaded campaign', campaign._campaign_dir)
    print('========================================================')
    sampler = campaign.get_active_sampler()
    sampler.load_state(os.path.join(work_dir_adapt, "campaign_sampler.pickle"))
    campaign.set_sampler(sampler)
    analysis = uq.analysis.SCAnalysis(sampler=sampler, qoi_cols=output_columns)
    analysis.load_state(os.path.join(
        work_dir_adapt, "campaign_analysis.pickle"))

    # look-ahead step (compute the code at admissible forward points)
    sampler.look_ahead(analysis.l_norm)

    # proceed as usual
    campaign.draw_samples()
    run_ids = campaign.populate_runs_dir()

    # copy generated run folders to SWEEP directory in config folder
    # 1. clean config SWEEP dir
    # 2. copy all generated runs by easyvvuq to config SWEEP folder
    #   2.1 only new run_id generated at this step should be copied
    path_to_config = find_config_file_path(config)
    sweep_dir = path_to_config + "/SWEEP"

    if os.path.exists(sweep_dir):
        rmtree(sweep_dir)
    os.mkdir(sweep_dir)
    print('=' * 20)
    print("Copying easyvvuq runs to %s SWEEP folder ..." % (config))
    for run_id in run_ids:
        with hide('output', 'running', 'warnings'), settings(warn_only=True):
            local("cp -r %s %s" % (os.path.join(campaign.work_dir,
                                                campaign.campaign_dir,
                                                'SWEEP',
                                                run_id
                                                ),
                                   os.path.join(sweep_dir,
                                                run_id
                                                )
                                   )
                  )
    print("Done")
    print('=' * 20)

    # submit ensemble jobs to remote machine
    # run the UQ ensemble at the admissible forward points
    job_label = campaign._campaign_dir

    if mode == 'serial':
        flee_script = 'flee'
    else:
        flee_script = 'pflee'

    # submit ensemble jobs to remote machine
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

    # save campaign and sampler
    campaign.save_state(os.path.join(work_dir_adapt, "campaign_state.json"))
    sampler.save_state(os.path.join(work_dir_adapt, "campaign_sampler.pickle"))

    backup_campaign_files()


@task
def flee_adapt_analyse(config, ** args):
    '''
    ============================================================================

        fab <remote_machine> flee_adapt_analyse:<conflict_name>
    example:

        fab eagle_vecma flee_adapt_analyse:mali

    ============================================================================
    '''

    load_campaign_files()

    # reload Campaign, sampler, analysis
    campaign = uq.Campaign(state_file=os.path.join(work_dir_adapt,
                                                   "campaign_state.json"),
                           work_dir=work_dir_adapt
                           )
    print('========================================================')
    print('Reloaded campaign', campaign._campaign_dir)
    print('========================================================')

    sampler = campaign.get_active_sampler()
    sampler.load_state(os.path.join(work_dir_adapt, "campaign_sampler.pickle"))
    campaign.set_sampler(sampler)

    # fetch only the required folder from remote machine
    with_config(config)
    # env.job_name_template += "_{}".format(job_label)
    # fetch results from remote machine
    job_label = campaign._campaign_dir
    job_folder_name = template(env.job_name_template + "_{}".format(job_label))
    fetch_results(regex=job_folder_name)

    # copy only output folder into local campaign_dir :)
    src = os.path.join(env.local_results, job_folder_name, 'RUNS')
    des = os.path.join(work_dir_adapt, campaign._campaign_dir, 'SWEEP')

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

    # Post-processing analysis
    analysis = uq.analysis.SCAnalysis(
        sampler=campaign._active_sampler,
        qoi_cols=output_columns
    )

    campaign.apply_analysis(analysis)

    # save analysis state
    # this is a temporary subroutine which saves the entire state of
    # the analysis in a pickle file. The proper place for this is the database
    analysis.save_state(os.path.join(
        work_dir_adapt, "campaign_analysis.pickle"))

    backup_campaign_files()


@task
def flee_adapt_init(config, simulation_period, mode='parallel', ** args):
    '''
    ============================================================================

        fab <remote_machine> flee_adapt_init:<conflict_name>,simulation_period=<number>

    example:

        fab eagle_vecma flee_adapt_init:mali,simulation_period=100

    ============================================================================
    '''

    # delete work_dir is exists
    if os.path.exists(work_dir_adapt):
        rmtree(work_dir_adapt)
    os.mkdir(work_dir_adapt)

    # Set up a fresh campaign called "flee-adaptive"
    campaign = uq.Campaign(name='flee-adaptive',
                           work_dir=work_dir_adapt)

    # to make sure we are not overwriting the new simulation on previous ones
    job_label = campaign._campaign_dir

    # Define parameter space for the flee-adaptive app
    params = json.load(open(os.path.join(get_plugin_path("FabFlee"),
                                         'templates',
                                         'params.json'
                                         )
                            )
                       )

    output_filename = params["out_file"]["default"]

    # Create an encoder and decoder
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

    # Add the flee-adaptive app
    campaign.add_app(name="flee-adaptive",
                          params=params,
                          encoder=encoder,
                          decoder=decoder,
                          collater=collater)

    # Set the active app to be flee-adaptive
    campaign.set_app("flee-adaptive")

    # parameters to vary
    vary = {
        "max_move_speed": cp.Uniform(100, 500),
        "max_walk_speed": cp.Uniform(10, 100),
        "camp_move_chance": cp.Uniform(0.0, 0.1),
        "conflict_move_chance": cp.Uniform(0.1, 1.0),
        #"default_move_chance": cp.Uniform(0.1, 1.0),
        #"camp_weight": cp.Uniform(1.0, 10.0),
        #"conflict_weight": cp.Uniform(0.1, 1.0)
    }

    #=================================
    # create dimension-adaptive sampler
    #=================================
    # sparse = use a sparse grid (required)
    # growth = use a nested quadrature rule (not required)
    # midpoint_level1 = use a single collocation point in the 1st iteration (not required)
    # dimension_adaptive = use a dimension adaptive sampler (required)
    sampler = uq.sampling.SCSampler(vary=vary,
                                    polynomial_order=1,
                                    quadrature_rule="C",
                                    sparse=True,
                                    growth=True,
                                    midpoint_level1=True,
                                    dimension_adaptive=True
                                    )

    campaign.set_sampler(sampler)
    campaign.draw_samples()
    run_ids = campaign.populate_runs_dir()

    # copy generated run folders to SWEEP directory in config folder
    # 1. clean config SWEEP dir
    # 2. copy all generated runs by easyvvuq to config SWEEP folder
    #   2.1 only new run_id generated at this step should be copied
    path_to_config = find_config_file_path(config)
    sweep_dir = path_to_config + "/SWEEP"

    if os.path.exists(sweep_dir):
        rmtree(sweep_dir)
    os.mkdir(sweep_dir)
    print('=' * 20)
    print("Copying easyvvuq runs to %s SWEEP folder ..." % (config))
    for run_id in run_ids:
        with hide('output', 'running', 'warnings'), settings(warn_only=True):
            local("cp -r %s %s" % (os.path.join(campaign.work_dir,
                                                campaign.campaign_dir,
                                                'SWEEP',
                                                run_id
                                                ),
                                   os.path.join(sweep_dir,
                                                run_id
                                                )
                                   )
                  )
    print("Done")
    print('=' * 20)

    if mode == 'serial':
        flee_script = 'flee'
    else:
        flee_script = 'pflee'

    # submit ensemble jobs to remote machine
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
    campaign.save_state(os.path.join(work_dir_adapt, "campaign_state.json"))
    sampler.save_state(os.path.join(work_dir_adapt, "campaign_sampler.pickle"))
    backup_campaign_files()


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
            {}/  {} ".format(work_dir_adapt, backup_dir)
        )


def load_campaign_files():

    with hide('output', 'running', 'warnings'), settings(warn_only=True):
        local(
            "rsync -av -m -v \
            --include='*.db' \
            --include='*.pickle' \
            --include='*.json' \
            --exclude='*' \
            {}/  {} ".format(backup_dir, work_dir_adapt)
        )
