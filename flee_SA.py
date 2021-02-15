from base.fab import *
import chaospy as cp
import numpy as np
import easyvvuq as uq
import matplotlib.pyplot as plt
import pandas
from pandas import DataFrame
from scipy.stats.mstats import gmean
import scipy.stats as st
import sys
import os
import math
import ruamel.yaml
from shutil import copyfile, rmtree
from pprint import pprint
import json
from plugins.FabFlee.FabFlee import *

# authors: Hamid Arabnejad, Diana Suleimenova, Wouter Edeling, Derek Groen


@task
@load_plugin_env_vars("FabFlee")
def flee_run_mcmc(config, simulation_period, mode='serial',
                  ** args):
    """
    ==========================================================================

    fab <remote_machine> flee_run_mcmc:<conflict_name>,simulation_period=<N>


    example:

        fab localhost flee_run_mcmc:mali,simulation_period=100

    Note : currently, we only support localhost execution
    ==========================================================================
    """
    if env['machine_name'] != 'localhost':
        raise RuntimeError("\ncurrently, only localhost is supported for "
                           "executing mcmc sampler.\n")
        exit()
    print('Start EasyVVUQ-MCMC ...')

    work_dir = os.path.join(os.path.dirname(__file__))

    update_environment()
    with_config(config)

    #############################################
    # load flee SA configuration from yml file #
    #############################################
    campaign_config = load_SA_campaign_config()
    MCMCSampler_setting = campaign_config['MCMCSampler_setting']

    campaign_name = 'flee_MCMC'
    campaign_work_dir = os.path.join(get_plugin_path('FabFlee'), campaign_name)

    ######################################
    # delete campaign_work_dir is exists #
    ######################################
    if os.path.exists(campaign_work_dir):
        rmtree(campaign_work_dir)
    os.makedirs(campaign_work_dir)

    ###########################
    # Set up a fresh campaign #
    ###########################
    db_location = "sqlite:///" + campaign_work_dir + "/campaign.db"
    campaign = uq.Campaign(name=campaign_name,
                           db_location=db_location,
                           work_dir=campaign_work_dir)

    #################################
    # Create an encoder and decoder #
    #################################
    campaign_config['decoder_output_column'] = 'Total error distribution'
    encoder = uq.encoders.GenericEncoder(
        template_fname=os.path.join(get_plugin_path('FabFlee'),
                                    'templates',
                                    campaign_config['encoder_template_fname']
                                    ),
        delimiter=campaign_config['encoder_delimiter'],
        target_filename=campaign_config['encoder_target_filename']
    )

    decoder = uq.decoders.SimpleCSV(
        target_filename=campaign_config['params']['out_file']['default'],
        output_columns=[campaign_config['decoder_output_column']]
    )

    ##########################
    # Add chain_id to params #
    ##########################
    campaign_config['params'].update({
        "chain_id": {"type": "integer", "default": 0}
    })

    ################################
    # Add the flee-SA-Sampler app #
    ################################
    campaign.add_app(name=campaign_name,
                     params=campaign_config['params'],
                     encoder=encoder,
                     decoder=decoder)

    ######################
    # parameters to vary #
    ######################
    vary_init = MCMCSampler_setting['vary_init']
    n_chains = None
    # Extra checking : To make sure all vary parameters has same array length
    for param_name, init_list in vary_init.items():
        if n_chains is None:
            n_chains = len(init_list)
        elif len(init_list) != n_chains:
            raise ValueError(
                "\nMake sure all parameters has same array length "
                "of initial values. Please check:"
                "\n\nMCMCSampler_setting:\n\t...\n\tvary_init:\n\t\t..."
                "\n\t\t%s : [...]"
                "\n\nin flee_SA_config.yml file" % (param_name))
            exit()

    ##########################################################
    # A function of one argument X (dictionary) that returns #
    # the proposal distribution conditional on  the X        #
    ##########################################################
    def distribution_func(x, b=1, var_names=vary_init.keys()):
        return cp.J(
            *(cp.Normal(x[var], b) for var in var_names)
        )

    output_column = campaign_config['decoder_output_column']
    sampler = uq.sampling.MCMCSampler(vary_init,
                                      distribution_func,
                                      output_column,
                                      n_chains=n_chains
                                      )
    ###########################################
    # Associate the sampler with the campaign #
    ###########################################
    campaign.set_sampler(sampler)

    #############################
    # Run EasyVVUQ MCMC sampler #
    #############################
    path_to_config = find_config_file_path(config)
    config_sweep_dir = os.path.join(path_to_config, 'SWEEP')

    if 'replicas_number' in MCMCSampler_setting:
        env.replicas = MCMCSampler_setting['replicas_number']
    else:
        env.replicas = 1

    for iter in range(MCMCSampler_setting['iteration_number']):
        ############################
        # Populate run directories #
        ############################
        campaign.draw_samples()
        runs_id = campaign.populate_runs_dir()
        print("runs_id : [%s]" % (','.join(runs_id)))

        ##################################################################
        # copy generated run folders to SWEEP directory in config folder #
        ##################################################################
        print("Copying new generated runs [%s] to config SWEEP folder ..." % (
            ','.join(runs_id))
        )
        with hide('output', 'running'):
            local("rm -rf %s/*" % (config_sweep_dir))
            # campaign2ensemble copies all run_ids, we don't need it here,
            # only new run_id generated at this step should be copied
            for run_id in runs_id:
                local("cp -r %s %s" % (os.path.join(campaign.work_dir,
                                                    campaign.campaign_dir,
                                                    'runs',
                                                    run_id
                                                    ),
                                       os.path.join(config_sweep_dir,
                                                    run_id
                                                    )
                                       )
                      )
        print('Done ...')

        ##########################################
        # submit ensemble jobs to remote machine #
        ##########################################
        job_label = campaign._campaign_dir
        if mode == 'serial':
            flee_script = 'flee'
        else:
            flee_script = 'pflee'

        env.script = flee_script
        env.simulation_period = simulation_period
        env.job_desc = "_mcmc"
        env.prevent_results_overwrite = 'delete'

        #########################################
        # runs to be executed in this iteration #
        #########################################
        # TODO: This should be returned by run_ensemble function.
        #       make sure run_ensemble
        #       returns the submit jobs folder name (not the full PATH)
        #       For now, they can be generated manually
        submited_runs_id = ["%s_%d" % (run_id, rep + 1)
                            for run_id in runs_id
                            for rep in range(env.replicas)
                            ]

        run_ensemble(config, config_sweep_dir, **args)

        #######################################################################
        # fetching results from remote machine                                #
        #     - here, instead of checking job scheduler, every time_step      #
        #       per each step, we fetch results and check if all the required #
        #       output files are available, which means all jobs are          #
        #       finished correctly                                            #
        #######################################################################
        job_folder_name = template(env.job_name_template)
        print(job_folder_name)

        time_step = 5  # minutes
        time_step_passed = 0

        while True:
            print("Fetching results from remote machine ...")
            with hide('output', 'running'), settings(warn_only=True):
                fetch_results(regex=job_folder_name)
            print("Done\n")

            unfinished_jobs = []
            for run_id in submited_runs_id:
                file_to_check = os.path.join(
                    env.local_results,
                    job_folder_name,
                    'RUNS',
                    run_id,
                    campaign_config['params']['out_file']['default']
                )
                print('file_to_check = %s' % (file_to_check))
                if not os.path.isfile(file_to_check):
                    unfinished_jobs.append(file_to_check)

            print("number of unfinished_jobs = %d" % (len(unfinished_jobs)))
            # print(unfinished_jobs)

            # wait for next round check
            if len(unfinished_jobs) == 0:
                break
            else:
                sleep(time_step * 60)
            time_step_passed += time_step

        print('Fetched all results from remote machine ...')

        #####################################################
        # Goes through all the submited_runs_id directories #
        # and calculates the validation scores              #
        #####################################################
        for run_id in runs_id:
            items = ["%s_%d" % (run_id, rep + 1)
                     for rep in range(env.replicas)]
            results_dir = "{}/{}/RUNS".format(env.local_results,
                                              job_folder_name)

            ensemble_vvp_results = vvp.ensemble_vvp(results_dir,
                                                    vvp_validate_results,
                                                    best_fit_distribution,
                                                    items=items
                                                    )

            print(">-" * 50)
            pprint(ensemble_vvp_results)
            print("<-" * 50)
            (_, run_id_data), *_ = ensemble_vvp_results.items()
            #######################################
            # Generate out.csv for each MCMC run #
            #######################################
            output_csv = os.path.join(campaign_work_dir,
                                      campaign._campaign_dir,
                                      'runs',
                                      run_id,
                                      'out.csv')
            dist_name = run_id_data['scores_aggregation']['dist_name']
            Total_error_distribution = run_id_data[
                'scores_aggregation']['value']
            # campaign_config['decoder_output_column']
            csv_headers = ['dist_name', 'Total error distribution']

            with open(output_csv, 'w') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=csv_headers)
                writer.writeheader()
                writer.writerow({
                    'dist_name': dist_name,
                    'Total error distribution': Total_error_distribution
                })

        #####################################################################
        # Combine the output from all runs associated with the current app #
        #####################################################################
        print('Run campaign.collate()')
        campaign.collate()
        ########################################################
        # Performs the MCMC sampling procedure on the campaign #
        ########################################################
        print('Run sampler.update()')
        sampler.update(campaign)


def best_fit_distribution(data, **kwargs):
    dist_names = ['gamma', 'beta', 'rayleigh', 'norm',
                  'pareto', 'laplace', 'uniform']
    data = np.array(data)
    distributions = []
    # Maximum Likelihood Estimation (MLE)
    mles = []
    for dist_name in dist_names:
        dist = getattr(st, dist_name)
        distributions.append(dist)
        pars = dist.fit(data)
        mle = dist.nnlf(pars, data)
        mles.append(mle)

    results = [(distribution.name, mle)for distribution, mle in
               zip(distributions, mles)]
    best_fit = sorted(zip(distributions, mles), key=lambda d: d[1])[0]

    print("input data = ")
    print(data)
    print('Best fit reached using {}, MLE value: {}'.format(
        best_fit[0].name, best_fit[1]))

    return {'dist_name': best_fit[0].name, 'value': best_fit[1]}


@task
@load_plugin_env_vars("FabFlee")
def flee_init_SA(config, simulation_period=-1, mode='serial',
                 sampler_name=None, ** args):
    """
    ==========================================================================

    fab <remote_machine> flee_init_SA:<conflict_name>,simulation_period=<N>

    example:

        fab eagle_vecma flee_init_SA:mali,simulation_period=100
        fab localhost flee_init_SA:mali,simulation_period=100

    ==========================================================================
    """
    update_environment()

    #############################################
    # load flee SA configuration from yml file #
    #############################################
    SA_campaign_config = load_SA_campaign_config()

    polynomial_order = SA_campaign_config['polynomial_order']
    if sampler_name is None:
        sampler_name = SA_campaign_config['sampler_name']

    campaign_name = 'flee_SA_%s' % (sampler_name)

    campaign_work_dir = os.path.join(get_plugin_path('FabFlee'),
                                     'flee_SA_%s' % (sampler_name)
                                     )

    runs_dir, campaign_dir = init_SA_campaign(campaign_name,
                                              SA_campaign_config,
                                              polynomial_order,
                                              campaign_work_dir)

    #############################################################
    # copy the EasyVVUQ campaign run set TO config SWEEP folder #
    #############################################################
    campaign2ensemble(config, campaign_dir)

    ###########################################################
    # set job_desc to avoid overwriting with previous SA jobs #
    ###########################################################
    env.job_desc = "_SA_%s" % (sampler_name)
    env.prevent_results_overwrite = 'delete'
    with_config(config)
    execute(put_configs, config)

    ##################################################
    # prepare env variable to submit an ensemble job #
    ##################################################
    if mode == 'serial':
        flee_script = 'flee'
    else:
        flee_script = 'pflee'

    ##########################################
    # submit ensemble jobs to remote machine #
    ##########################################
    flee_ensemble(config,
                  simulation_period,
                  script=flee_script,
                  **args)


@task
@load_plugin_env_vars("FabFlee")
def flee_analyse_SA(config, sampler_name=None, ** args):
    """
    ==========================================================================

        fab <remote_machine> flee_analyse_SA:<conflict_name>

    example:

        fab eagle_vecma flee_analyse_SA:mali
        fab localhost flee_analyse_SA:mali

    ==========================================================================
    """
    update_environment()

    #############################################
    # load flee SA configuration from yml file #
    #############################################
    SA_campaign_config = load_SA_campaign_config()

    polynomial_order = SA_campaign_config['polynomial_order']
    if sampler_name is None:
        sampler_name = SA_campaign_config['sampler_name']

    campaign_name = 'flee_SA_%s' % (sampler_name)

    campaign_work_dir = os.path.join(get_plugin_path('FabFlee'),
                                     'flee_SA_%s' % (sampler_name)
                                     )

    load_campaign_files(campaign_work_dir)

    ###################
    # reload Campaign #
    ###################
    campaign = uq.Campaign(state_file=os.path.join(campaign_work_dir,
                                                   'campaign_state.json'),
                           work_dir=campaign_work_dir
                           )
    print('===========================================')
    print('Reloaded campaign', campaign._campaign_dir)
    print('===========================================')

    sampler = campaign.get_active_sampler()
    campaign.set_sampler(sampler)

    ####################################################
    # fetch results from remote machine                #
    # here, we ONLY fetch the required results folders #
    ####################################################
    env.job_desc = "_SA_%s" % (sampler_name)
    with_config(config)

    job_folder_name = template(env.job_name_template)
    print("fetching results from remote machine ...")
    with hide('output', 'running', 'warnings'), settings(warn_only=True):
        fetch_results(regex=job_folder_name)
    print("Done\n")

    #####################################################
    # copy ONLY the required output files for analyse,  #
    # i.e., EasyVVUQ.decoders.target_filename           #
    #####################################################
    target_filename = SA_campaign_config['params']['out_file']['default']
    src = os.path.join(env.local_results, job_folder_name, 'RUNS')
    des = campaign.campaign_db.runs_dir()
    print("Syncing output_dir ...")
    # with hide('output', 'running', 'warnings'), settings(warn_only=True):
    local(
        "rsync -pthrz "
        "--include='/*/' "
        "--include='{}' "
        "--exclude='*' "
        "{}/  {} ".format(target_filename, src, des)
    )
    print("Done ...\n")

    ##################################################
    # save dataframe containing all collated results #
    ##################################################
    campaign.collate()
    collation_result = campaign.get_collation_result()
    collation_result.to_csv(os.path.join(campaign_work_dir,
                                         'collation_result.csv'
                                         ),
                            index=False
                            )

    ###################################
    #    Post-processing analysis     #
    ###################################
    output_column = SA_campaign_config['decoder_output_column']

    if sampler_name == 'SCSampler':
        analysis = uq.analysis.SCAnalysis(
            sampler=campaign._active_sampler,
            qoi_cols=[output_column])
    elif sampler_name == 'PCESampler':
        analysis = uq.analysis.PCEAnalysis(
            sampler=campaign._active_sampler,
            qoi_cols=[output_column])

    campaign.apply_analysis(analysis)
    results = campaign.get_last_analysis()

    #########################################################
    # save everything, campaign, sampler and analysis state #
    #########################################################
    campaign.save_state(os.path.join(campaign_work_dir,
                                     "campaign_state.json"))

    ###################################
    #    Plot statistical_moments     #
    ###################################
    mean = results.describe()[output_column].loc[['mean']].values.ravel()
    std = results.describe()[output_column].loc[['std']].values.ravel()
    X = range(len(mean))
    fig = plt.figure()
    ax = fig.add_subplot(111, xlabel="days",
                         ylabel="velocity %s" % (output_column),
                         title="code mean +/- standard deviation")
    ax.plot(X, mean, 'b-', label='mean')
    ax.plot(X, mean - std, '--r', label='+1 std-dev')
    ax.plot(X, mean + std, '--r')
    ax.fill_between(X, mean - std, mean + std, color='r', alpha=0.2)
    plt.tight_layout()
    plt.legend(loc='best')
    plot_file_name = 'plot_statistical_moments[%s]' % (output_column)
    plt.savefig(os.path.join(campaign_work_dir, plot_file_name),
                dpi=400)

    ###################################
    #        Plot sobols_first        #
    ###################################
    fig = plt.figure()
    ax = fig.add_subplot(111, xlabel="days",
                         ylabel="Sobol indices",
                         title="First order Sobol index")
    ax.set_title("First order Sobol index [output column = %s]"
                 % (output_column),
                 fontsize=10, fontweight='bold', loc='center')
    sobols_first = results.raw_data["sobols_first"][output_column]
    param_i = 0
    for v in sobols_first:
        y = sobols_first[v].ravel()
        important = False
        if y[-1] != 0:
            ax.plot(y, label=v)
        else:
            print("%s ignored for plotting" % (v))

        param_i = param_i + 1

    plt.legend(loc='best')
    plt.tight_layout()
    plot_file_name = 'plot_sobols_first[%s]' % (output_column)
    plt.savefig(os.path.join(campaign_work_dir, plot_file_name),
                dpi=400)

    ###############################################################
    # yml_results contains all campaign info and analysis results #
    # it will be saved in sobols.yml file                         #
    ###############################################################
    S = ruamel.yaml.scalarstring.DoubleQuotedScalarString
    yml_results = ruamel.yaml.comments.CommentedMap()
    yml_results.update({'campaign_info': {}})
    yml_results['campaign_info'].update({
        'name': S(campaign._active_app_name),
        'work_dir': S(campaign.work_dir),
        'num_runs': campaign.campaign_db.get_num_runs(),
        'output_column': S(output_column),
        'polynomial_order': polynomial_order,
        'sampler': S(SA_campaign_config['sampler_name']),
        'distribution_type': S(SA_campaign_config['distribution_type']),
        'sparse': S(SA_campaign_config['sparse']),
        'growth': S(SA_campaign_config['growth'])
    })
    if sampler_name == 'SCSampler':
        yml_results['campaign_info'].update({
            'quadrature_rule': S(SA_campaign_config['quadrature_rule']),
            'midpoint_level1': S(SA_campaign_config['midpoint_level1']),
            'dimension_adaptive': S(SA_campaign_config['dimension_adaptive'])
        })
    elif sampler_name == 'PCESampler':
        yml_results['campaign_info'].update({
            'rule': S(SA_campaign_config['quadrature_rule']),
        })

    ROUND_NDIGITS = 4
    for param in SA_campaign_config['selected_vary_parameters']:
        # I used CommentedMap for adding comments
        yml_results[param] = ruamel.yaml.comments.CommentedMap()
        # yml_results.update({param: {}})
        yml_results[param].update({
            "sobols_first_mean":
                round(float(np.mean(sobols_first[param].ravel())),
                      ROUND_NDIGITS),
            "sobols_first_gmean":
                round(float(gmean(sobols_first[param].ravel())),
                      ROUND_NDIGITS),
            "sobols_first":
                np.around(sobols_first[param].ravel(),
                          ROUND_NDIGITS).tolist()
        })
        yml_results[param].yaml_set_comment_before_after_key(
            "sobols_first_gmean",
            before="geometric mean, i.e., n-th root of (x1 * x2 * … * xn)",
            indent=2)
        yml_results[param].yaml_set_comment_before_after_key(
            "sobols_first_mean",
            before="arithmetic mean i.e., (x1 + x2 + … + xn)/n",
            indent=2)

    yaml = ruamel.yaml.YAML()
    yaml.preserve_quotes = True
    yaml.default_flow_style = None
    # to Prevent long lines getting wrapped in ruamel.yaml
    # we set the yaml.width to a big enough value to prevent line-wrap
    yaml.width = sys.maxsize

    res_file_name = os.path.join(campaign_work_dir, 'sobols.yml')
    print(res_file_name)
    with open(res_file_name, 'w') as outfile:
        yaml.dump(yml_results, outfile)


def init_SA_campaign(campaign_name, campaign_config,
                     polynomial_order, campaign_work_dir):

    ######################################
    # delete campaign_work_dir is exists #
    ######################################
    if os.path.exists(campaign_work_dir):
        rmtree(campaign_work_dir)
    os.makedirs(campaign_work_dir)

    ###########################
    # Set up a fresh campaign #
    ###########################
    db_location = "sqlite:///" + campaign_work_dir + "/campaign.db"
    campaign = uq.Campaign(name=campaign_name,
                           db_location=db_location,
                           work_dir=campaign_work_dir)

    #################################
    # Create an encoder and decoder #
    #################################
    encoder = uq.encoders.GenericEncoder(
        template_fname=os.path.join(get_plugin_path('FabFlee'),
                                    'templates',
                                    campaign_config['encoder_template_fname']
                                    ),
        delimiter=campaign_config['encoder_delimiter'],
        target_filename=campaign_config['encoder_target_filename']
    )

    decoder = uq.decoders.SimpleCSV(
        target_filename=campaign_config['params']['out_file']['default'],
        output_columns=[campaign_config['decoder_output_column']]
    )

    ################################
    # Add the flee-SA-Sampler app #
    ################################
    campaign.add_app(name=campaign_name,
                     params=campaign_config['params'],
                     encoder=encoder,
                     decoder=decoder)

    ######################
    # parameters to vary #
    ######################
    vary = {}
    for param in campaign_config['selected_vary_parameters']:
        pprint(campaign_config[
            'vary_parameters_range'][param])
        lower_value = campaign_config[
            'vary_parameters_range'][param]['range'][0]
        upper_value = campaign_config[
            'vary_parameters_range'][param]['range'][1]
        if campaign_config['distribution_type'] == 'DiscreteUniform':
            vary.update({param: cp.DiscreteUniform(lower_value, upper_value)})
        elif campaign_config['distribution_type'] == 'Uniform':
            vary.update({param: cp.Uniform(lower_value, upper_value)})

    ####################
    # create Sampler #
    ####################
    if campaign_config['sampler_name'] == 'SCSampler':
        sampler = uq.sampling.SCSampler(
            vary=vary,
            polynomial_order=polynomial_order,
            quadrature_rule=campaign_config['quadrature_rule'],
            growth=campaign_config['growth'],
            sparse=campaign_config['sparse'],
            midpoint_level1=campaign_config['midpoint_level1'],
            dimension_adaptive=campaign_config['dimension_adaptive']
        )
    elif campaign_config['sampler_name'] == 'PCESampler':
        sampler = uq.sampling.PCESampler(
            vary=vary,
            polynomial_order=polynomial_order,
            rule=campaign_config['quadrature_rule'],
            sparse=campaign_config['sparse'],
            growth=campaign_config['growth']
        )
    # TODO: add other sampler here

    ###########################################
    # Associate the sampler with the campaign #
    ###########################################
    campaign.set_sampler(sampler)

    #########################################
    # draw all of the finite set of samples #
    #########################################
    campaign.draw_samples()
    run_ids = campaign.populate_runs_dir()

    ###################################
    # save campaign and sampler state #
    ###################################
    campaign.save_state(os.path.join(campaign_work_dir,
                                     "campaign_state.json"
                                     )
                        )

    print("=" * 50)
    print("With user's specified parameters for sampler")
    print("init_SA_campaign generates %d runs" % (len(run_ids)))
    print("in %s" % (campaign_work_dir))
    print("=" * 50)

    ######################################################
    # backup campaign files, i.e, *.db, *.json, *.pickle #
    ######################################################
    backup_campaign_files(campaign.work_dir)

    runs_dir = campaign.campaign_db.runs_dir()
    campaign_dir = campaign.campaign_db.campaign_dir()
    return runs_dir, campaign_dir


def load_SA_campaign_config():
    flee_SA_yml_PATH = os.path.join(get_plugin_path('FabFlee'),
                                    'flee_SA_config.yml')
    SA_campaign_config = yaml.load(open(flee_SA_yml_PATH),
                                   Loader=yaml.SafeLoader
                                   )
    #####################################################
    # load parameter space for the easyvvuq sampler app #
    #####################################################
    sampler_params_json_PATH = os.path.join(get_plugin_path('FabFlee'),
                                            'templates',
                                            'params.json'
                                            )
    sampler_params = json.load(open(sampler_params_json_PATH))

    #####################################################
    # add loaded campaign params to SA_campaign_config #
    #####################################################
    SA_campaign_config.update({'params': sampler_params})

    return SA_campaign_config


def backup_campaign_files(campaign_work_dir):
    backup_dir = os.path.join(campaign_work_dir, 'backup')
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
            {}/  {} ".format(campaign_work_dir, backup_dir)
        )


def load_campaign_files(campaign_work_dir):
    backup_dir = os.path.join(campaign_work_dir, 'backup')
    with hide('output', 'running', 'warnings'), settings(warn_only=True):
        local(
            "rsync -av -m -v \
            --include='*.db' \
            --include='*.pickle' \
            --include='*.json' \
            --exclude='*' \
            {}/  {} ".format(backup_dir, campaign_work_dir)
        )
