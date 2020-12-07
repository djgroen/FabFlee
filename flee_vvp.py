# This file contains the function definitions for Verification and Validation
# Patterns (VVP) specific to FabFlee.
# Pattern-1: Stable Intermediate Forms (SIF)
# Pattern-2: Level of Refinement (LoR)
# Pattern-3: Ensemble Output Validation (EoV)
# Pattern-4: Quantity of Interest (QoI)
#
# author: Hamid Arabnejad
#
from base.fab import *
from pprint import pprint
import yaml
import ruamel.yaml
import os
import json
from shutil import rmtree, copy2
import easyvvuq as uq
import chaospy as cp
import numpy as np
import pandas as pd
import re
import pickle
import matplotlib.pyplot as plt
from scipy.stats.mstats import gmean
from plugins.FabFlee.FabFlee import *
from VVP.vvp import ensemble_vvp_LoR


@task
@load_plugin_env_vars("FabFlee")
def flee_init_vvp_LoR(config, simulation_period, mode='serial', **args):
    """
    Level of Refinement (LoR) is a general verification pattern that seeks
    asymptotic behaviour in QoI upon increasing the resolution of certain
    model parameters. It is important to note that the same quantity of
    interest is computed at every given resolution.

    flee_init_vvp_LoR will submit a series of ensemble job
        for each ensemble job, different polynomial_order used to
        generated an EasyVVUQ campaign run set

    usage example:
        fab localhost flee_init_vvp_LoR:mali,simulation_period=10

    """
    update_environment()

    #############################################
    # load flee vvp configuration from yml file #
    #############################################
    vvp_campaign_config = load_vvp_campaign_config()

    polynomial_order_range = range(
        vvp_campaign_config['polynomial_order_range']['start'],
        vvp_campaign_config['polynomial_order_range']['end'],
        vvp_campaign_config['polynomial_order_range']['step']
    )
    sampler_name = vvp_campaign_config['sampler_name']

    for polynomial_order in polynomial_order_range:
        campaign_name = 'flee_vvp_LoR_%s_po%d_' % (
            sampler_name, polynomial_order)
        campaign_work_dir = os.path.join(get_plugin_path('FabFlee'),
                                         'flee_vvp_LoR_%s' % (sampler_name),
                                         'campaign_po%d' % (polynomial_order)
                                         )
        runs_dir, campaign_dir = init_vvp_campaign(campaign_name,
                                                   vvp_campaign_config,
                                                   polynomial_order,
                                                   campaign_work_dir)

        #############################################################
        # copy the EasyVVUQ campaign run set TO config SWEEP folder #
        #############################################################
        campaign2ensemble(config, campaign_dir)

        ############################################################
        # set job_desc to avoid overwriting with previous vvp jobs #
        ############################################################
        env.job_desc = "_vvp_LoR_%s_po%d" % (sampler_name, polynomial_order)
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
def flee_analyse_vvp_LoR(config):
    """
    flee_analyse_vvp_LoR will analysis the output of each vvp ensemble series

    usage example:
        fab localhost flee_analyse_vvp_LoR:mali
    """
    update_environment()

    #############################################
    # load flee vvp configuration from yml file #
    #############################################
    vvp_campaign_config = load_vvp_campaign_config()

    polynomial_order_range = range(
        vvp_campaign_config['polynomial_order_range']['start'],
        vvp_campaign_config['polynomial_order_range']['end'],
        vvp_campaign_config['polynomial_order_range']['step']
    )
    sampler_name = vvp_campaign_config['sampler_name']

    ###########################################
    # set a default dir to save results sobol #
    ###########################################
    sobol_work_dir = os.path.join(get_plugin_path('FabFlee'),
                                  'flee_vvp_LoR_%s' % (sampler_name),
                                  'sobol')

    ###################################
    # delete sobol_work_dir is exists #
    ###################################
    if os.path.exists(sobol_work_dir):
        rmtree(sobol_work_dir)
    os.makedirs(sobol_work_dir)

    for polynomial_order in polynomial_order_range:
        campaign_work_dir = os.path.join(get_plugin_path('FabFlee'),
                                         'flee_vvp_LoR_%s' % (sampler_name),
                                         'campaign_po%d' % (polynomial_order)
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
        env.job_desc = "_vvp_LoR_%s_po%d" % (sampler_name, polynomial_order)
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
        target_filename = vvp_campaign_config['params']['out_file']['default']
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
        collation_result.to_pickle(os.path.join(campaign_work_dir,
                                                'collation_result.pickle'
                                                ))
        ###################################
        #    Post-processing analysis     #
        ###################################
        output_column = vvp_campaign_config['decoder_output_column']

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
        '''
        sampler.save_state(os.path.join(campaign_work_dir,
                                        "campaign_sampler.pickle"))
        analysis.save_state(os.path.join(campaign_work_dir,
                                         "campaign_analysis.pickle"))
        '''
        ###################################
        #    Plot statistical_moments     #
        ###################################
        # ------------------------------------
        # here I use ravel to fix this issue |
        # SCSampler  : array([...])          |
        # PCESampler : array([[...]])        |
        # ------------------------------------
        fig_desc = 'polynomial_order = %d ,num_runs = %d' % (
            polynomial_order, campaign.campaign_db.get_num_runs())
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        fig, ax = plt.subplots()
        ax.set_xlabel('days')
        ax.set_ylabel("velocity %s" % (output_column))
        fig.suptitle("code mean +/- standard deviation\n",
                     fontsize=10, fontweight='bold')
        ax.set_title(fig_desc, fontsize=8, loc='center',
                     fontweight='bold', bbox=props)
        mean = results.describe()[output_column]['mean'].ravel()
        std = results.describe()[output_column]['std'].ravel()
        X = range(len(mean))
        ax.plot(X, mean, 'b-', label='mean')
        ax.plot(X, mean - std, '--r', label='+1 std-dev')
        ax.plot(X, mean + std, '--r')
        ax.fill_between(X, mean - std, mean + std, color='r', alpha=0.2)
        # plt.tight_layout()
        plt.legend(loc='best')
        plot_file_name = 'plot_statistical_moments[%s]' % (output_column)
        plt.savefig(os.path.join(campaign_work_dir, plot_file_name),
                    dpi=400)

        ###################################
        #        Plot sobols_first        #
        ###################################
        fig, ax = plt.subplots()
        ax.set_xlabel('days')
        ax.set_ylabel('Sobol indices')
        fig.suptitle("First order Sobol index [output column = %s]\n"
                     % (output_column),
                     fontsize=10, fontweight='bold')
        ax.set_title(fig_desc, fontsize=8, loc='center',
                     fontweight='bold', bbox=props)

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
        # plt.tight_layout()
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
            'sampler': S(vvp_campaign_config['sampler_name']),
            'distribution_type': S(vvp_campaign_config['distribution_type']),
            'sparse': S(vvp_campaign_config['sparse']),
            'growth': S(vvp_campaign_config['growth'])
        })
        if sampler_name == 'SCSampler':
            yml_results['campaign_info'].update({
                'quadrature_rule': S(vvp_campaign_config['quadrature_rule']),
                'midpoint_level1': S(vvp_campaign_config['midpoint_level1']),
                'dimension_adaptive': S(vvp_campaign_config
                                        ['dimension_adaptive'])
            })
        elif sampler_name == 'PCESampler':
            yml_results['campaign_info'].update({
                'rule': S(SA_campaign_config['quadrature_rule']),
            })

        ROUND_NDIGITS = 4
        for param in vvp_campaign_config['selected_vary_parameters']:
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
            # add comments to yml
            '''
            yml_results[param].yaml_add_eol_comment(
                "geometric mean, i.e.,  n-th root of (x1 * x2 * … * xn)",
                "sobols_first_gmean")
            yml_results[param].yaml_add_eol_comment(
                "arithmetic mean i.e., (x1 + x2 + … + xn)",
                "sobols_first_mean")
            '''
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
            '''
            yaml.dump(yml_results, outfile,
                      default_flow_style=None, width=1000)
            '''

        ########################################
        # copy sobols.yml file to sobol folder #
        ########################################
        print("copy sobols.yml file to sobol folder ...")
        # here instead of mkdirs and copy, I used rsync
        local(
            "rsync -pthrz "
            "--include='/*/' "
            "--include='sobols.yml' "
            "--include='*.png' "
            "--exclude='*' "
            "{}  {} ".format(campaign_work_dir, sobol_work_dir)
        )
        print("Done ...\n")

    #####################################################
    # Check the convergence of the SC Sobols indices    #
    # with polynomial refinement                        #
    #####################################################
    ensemble_vvp_LoR(results_dirs_PATH=sobol_work_dir,
                     load_QoIs_function=load_QoIs_function,
                     aggregation_function=plot_convergence,
                     plot_file_path=sobol_work_dir
                     )


def plot_convergence(scores, plot_file_path):
    """
    The VVP agregation_function, compares the sobol indices (as function of
    the polynomial order)
    input scores structure:
    result_dir_1_name:
        order: <polynomial_order>
        value:
            vary_param_1: {<sobol_func_name>:<value>}
            ...
            vary_param_z: {<sobol_func_name>:<value>}
    ...
    result_dir_N_name:
        order: <polynomial_order>
        value:
            vary_param_1: {<sobol_func_name>:<value>}
            ...
            vary_param_z: {<sobol_func_name>:<value>}

    ------------------------------------------------------------------
    NOTE: Here, we use the result with maximum polynomial order as the
        reference value
    """
    last_item_key = list(scores)[-1]

    #############################################
    # ref_sobols_value structure:               #
    #                                           #
    # vary_param_1: {<sobol_func_name>:<value>} #
    # ...                                       #
    # vary_param_n: {<sobol_func_name>:<value>} #
    #############################################

    ref_sobols_value = scores[last_item_key]['value']

    results = {}
    results.update({'polynomial_order': []})
    compare_res = {}
    for run_dir in scores:
        polynomial_order = scores[run_dir]['order']
        results['polynomial_order'].append(polynomial_order)
        poly_key = 'polynomial_order %d' % (polynomial_order)
        compare_res.update({poly_key: {}})
        for param in scores[run_dir]['value']:
            if param not in results:
                results.update({param: []})
            sb_func_name = list(scores[run_dir]['value'][param].keys())[0]
            sb = scores[run_dir]['value'][param][sb_func_name]
            results[param].append(sb)

    #############################################
    # plotting results                          #
    # results dict structure                    #
    #       vary_param_1: [run1,run2,...]       #
    #       vary_param_2: [run1,run2,...]       #
    #       polynomial_order: [po1,po2,...]     #
    #############################################

    params = list(results.keys())
    params.remove('polynomial_order')
    fig = plt.figure()
    ax = fig.add_subplot(111, xlabel="polynomial order",
                         ylabel="sobol indices",
                         title="convergence")
    X = range(len(results['polynomial_order']))
    for param in params:
        ax.plot(X, results[param], label=param)

    plt.xticks(X, results['polynomial_order'])
    plt.tight_layout()
    plt.legend(loc='best')
    convergence_plot_file_name = 'vvp_QoI_convergence.png'
    plt.savefig(os.path.join(plot_file_path, convergence_plot_file_name),
                dpi=400)
    print('=' * 50)
    print('The convergence plot generated ...')
    print(os.path.join(plot_file_path, convergence_plot_file_name))
    print('=' * 50)


def load_QoIs_function(result_dir):
    """
    we load input sobols.yml with this structure:
    vary_param_1:
        sobols_first: <array[....]>
        sobols_first_gmean: <value>
        sobols_first_mean: <value>
    ...
    vary_param_N:
        sobols_first: <array[....]>
        sobols_first_gmean: <value>
        sobols_first_mean: <value>
    campaign_info:
        distribution_type: <str>
        name: <str> # name of campaign
        num_runs: <int>
        polynomial_order: <int>
        sampler: <str> # name of sampler
        work_dir: <str> # PATH to this campaign result

    The returns values are : QoIs_values,polynomial_order
        In this implementation, QoIs_values has this
    vary_param_1:
        score: <value>
    ...
    vary_param_N:
        score: <value>
    """
    data_file_name = os.path.join(result_dir, 'sobols.yml')
    sobols_data = yaml.load(open(data_file_name), Loader=yaml.SafeLoader)
    polynomial_order = sobols_data['campaign_info']['polynomial_order']
    num_runs = sobols_data['campaign_info']['num_runs']
    del sobols_data['campaign_info']

    # sobols_first_mean or sobols_first_gmean
    score_column_name = 'sobols_first_mean'

    QoIs_values = {}
    for param in sobols_data:
        QoIs_values.update({param: {}})
        for key in sobols_data[param]:
            if key == score_column_name:
                QoIs_values[param].update({
                    key: sobols_data[param][key]
                })

    return QoIs_values, polynomial_order, num_runs


def load_vvp_campaign_config():

    flee_vvp_yml_PATH = os.path.join(get_plugin_path('FabFlee'),
                                     'flee_vvp_config.yml')
    vvp_campaign_config = yaml.load(open(flee_vvp_yml_PATH),
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
    # add loaded campaign params to vvp_campaign_config #
    #####################################################
    vvp_campaign_config.update({'params': sampler_params})

    return vvp_campaign_config


def init_vvp_campaign(campaign_name, campaign_config,
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
    # Add the flee-vvp-Sampler app #
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
    # TODO:	add other sampler here

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
    print("init_vvp_campaign generates %d runs" % (len(run_ids)))
    print("in %s" % (campaign_work_dir))
    print("=" * 50)

    ######################################################
    # backup campaign files, i.e, *.db, *.json, *.pickle #
    ######################################################
    backup_campaign_files(campaign.work_dir)

    runs_dir = campaign.campaign_db.runs_dir()
    campaign_dir = campaign.campaign_db.campaign_dir()
    return runs_dir, campaign_dir


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
