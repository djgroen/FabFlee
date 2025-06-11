#Documention: https://github.com/djgroen/FabFlee/blob/master/doc/TutorialSensitivity.md

try:
    from fabsim.base.fab import *
except ImportError:
    from base.fab import *

import chaospy as cp
import numpy as np
import easyvvuq as uq
import matplotlib.pyplot as plt
import pandas as pd
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
def flee_init_SA(config, simulation_period=-1, mode="serial",
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
    flee_SA_config_file = os.path.join(
        get_plugin_path("FabFlee"),
        "SA",
        "flee_SA_config.yml"
    )
    SA_campaign_config = load_SA_campaign_config(
        plugin_name="FabFlee", SA_config_file=flee_SA_config_file
    )

    polynomial_order = SA_campaign_config["polynomial_order"]
    if sampler_name is None:
        sampler_name = SA_campaign_config["sampler_name"]

    campaign_name = "flee_SA_{}_{}".format(sampler_name, config)

    campaign_work_dir = os.path.join(
        get_plugin_path("FabFlee"),
        "SA",
        "flee_SA_{}_{}".format(sampler_name, config)
    )

    runs_dir, campaign_dir = init_SA_campaign(
        plugin_name="FabFlee",
        campaign_name=campaign_name,
        campaign_config=SA_campaign_config,
        polynomial_order=polynomial_order,
        campaign_work_dir=campaign_work_dir
    )

    #############################################################
    # copy the EasyVVUQ campaign run set TO config SWEEP folder #
    #############################################################
    campaign2ensemble(config, campaign_dir)

    ###########################################################
    # set job_desc to avoid overwriting with previous SA jobs #
    ###########################################################
    env.job_desc = "_SA_%s" % (sampler_name)
    env.prevent_results_overwrite = "delete"
    with_config(config)
    execute(put_configs, config)

    ##################################################
    # prepare env variable to submit an ensemble job #
    ##################################################
    if mode == "serial":
        flee_script = "flee"
    else:
        flee_script = "pflee"

    ##########################################
    # submit ensemble jobs to remote machine #
    ##########################################
    flee_ensemble(
        config,
        simulation_period,
        script=flee_script,
        **args
    )


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
    flee_SA_config_file = os.path.join(
        get_plugin_path("FabFlee"),
        "SA",
        "flee_SA_config.yml"
    )
    SA_campaign_config = load_SA_campaign_config(
        plugin_name="FabFlee", SA_config_file=flee_SA_config_file
    )

    polynomial_order = SA_campaign_config["polynomial_order"]
    if sampler_name is None:
        sampler_name = SA_campaign_config["sampler_name"]

    campaign_name = "flee_SA_{}_{}".format(sampler_name, config)

    campaign_work_dir = os.path.join(
        get_plugin_path("FabFlee"),
        "SA",
        "flee_SA_{}_{}".format(sampler_name, config)
    )

    load_campaign_files(campaign_work_dir)

    ###################
    # reload Campaign #
    ###################
    db_location = "sqlite:///" + campaign_work_dir + "/campaign.db"
    campaign = uq.Campaign(name=campaign_name, db_location=db_location)
    print("===========================================")
    print("Reloaded campaign {}".format(campaign_name))
    print("===========================================")

    sampler = campaign.get_active_sampler()
    campaign.set_sampler(sampler, update=True)

    ####################################################
    # fetch results from remote machine                #
    # here, we ONLY fetch the required results folders #
    ####################################################
    env.job_desc = "_SA_{}".format(sampler_name)
    with_config(config)

    job_folder_name = template(env.job_name_template)
    print("fetching results from remote machine ...")
    fetch_results(regex=job_folder_name, files="out.csv")
    print("Done\n")

    #####################################################
    # copy ONLY the required output files for analyse,  #
    # i.e., EasyVVUQ.decoders.target_filename           #
    #####################################################
    output_filename = SA_campaign_config["params"]["out_file"]["default"]
    src = os.path.join(env.local_results, job_folder_name, "RUNS")
    des = campaign.campaign_db.runs_dir()
    print("Syncing output_dir ...")
    local(
        "rsync -pthrz "
        "--include='/*/' "
        "--include='{}' "
        "--exclude='*' "
        "{}/  {} ".format(output_filename, src, des)
    )
    print("Done ...\n")

    #######################################
    # Create an decoder for data analysis #
    #######################################
    output_column = SA_campaign_config["decoder_output_column"]
    decoder = uq.decoders.SimpleCSV(
        target_filename=output_filename,
        output_columns=[output_column]
    )

    #####################
    # execute collate() #
    #####################
    actions = uq.actions.Actions(
        uq.actions.Decode(decoder)
    )
    campaign.replace_actions(campaign_name, actions)
    campaign.execute().collate()
    collation_result = campaign.get_collation_result()

    ##################################################
    # save dataframe containing all collated results #
    ##################################################
    collation_result.to_csv(
        os.path.join(campaign_work_dir, "collation_result.csv"),
        index=False
    )

    ###################################
    #    Post-processing analysis     #
    ###################################

    if sampler_name == "SCSampler":
        analysis = uq.analysis.SCAnalysis(
            sampler=campaign._active_sampler,
            qoi_cols=[output_column]
        )
    elif sampler_name == "PCESampler":
        analysis = uq.analysis.PCEAnalysis(
            sampler=campaign._active_sampler,
            qoi_cols=[output_column]
        )

    campaign.apply_analysis(analysis)
    results = campaign.get_last_analysis()

    ###################
    #    Plotting     #
    ###################
    fig_desc = "polynomial_order = {}, num_runs = {}, sampler = {}".format(
        polynomial_order,
        campaign.campaign_db.get_num_runs(),
        sampler_name
    )
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)

    ########################
    #    Plot raw data     #
    ########################
    output_files = glob.glob(
        os.path.join(campaign_work_dir + "/**/%s" % (output_filename)),
        recursive=True
    )

    fig, ax = plt.subplots()
    ax.set_xlabel("days")
    ax.set_ylabel(output_column)
    fig.suptitle(
        "RAW data : column {}\n".format(output_column),
        fontsize=10, fontweight="bold"
    )
    ax.set_title(
        fig_desc, fontsize=8, loc="center",
        fontweight="bold", bbox=props
    )
    for output_file in output_files:
        total_errors = pd.read_csv(output_file)[
            output_column].values.tolist()
        ax.plot(total_errors)

    plot_file_name = "raw[{}]".format(output_column)
    plt.savefig(os.path.join(campaign_work_dir, plot_file_name),
                dpi=400)

    ###################################
    #    Plot statistical_moments     #
    ###################################
    fig, ax = plt.subplots()
    ax.set_xlabel("days")
    ax.set_ylabel("velocity {}".format(output_column))
    fig.suptitle(
        "code mean +/- standard deviation\n",
        fontsize=10, fontweight="bold"
    )
    ax.set_title(
        fig_desc, fontsize=8, loc="center",
        fontweight="bold", bbox=props
    )

    mean = results.describe(output_column, "mean")
    std = results.describe(output_column, "std")
    X = range(len(mean))
    ax.plot(X, mean, "b-", label="mean")
    ax.plot(X, mean - std, "--r", label="+1 std-dev")
    ax.plot(X, mean + std, "--r")
    ax.fill_between(X, mean - std, mean + std, color="r", alpha=0.2)
    # plt.tight_layout()
    plt.legend(loc="best")
    plot_file_name = "plot_statistical_moments[{}]".format(output_column)
    plt.savefig(os.path.join(campaign_work_dir, plot_file_name),
                dpi=400)

    ###################################
    #        Plot sobols_first        #
    ###################################
    fig, ax = plt.subplots()
    ax.set_xlabel("days")
    ax.set_ylabel("Sobol indices")
    fig.suptitle(
        "First order Sobol index [output column = {}]\n".format(
            output_column),
        fontsize=10, fontweight="bold"
    )
    ax.set_title(
        fig_desc, fontsize=8, loc='center',
        fontweight='bold', bbox=props
    )

    # sobols_first = results.raw_data["sobols_first"][output_column]
    sobols_first = results.sobols_first(output_column)
    param_i = 0
    for v in sobols_first:
        y = sobols_first[v].ravel()
        important = False
        if y[-1] != 0:
            ax.plot(y, label=v)
        else:
            print("{} ignored for plotting".format(v))

        param_i = param_i + 1

    plt.legend(loc="best")
    # plt.tight_layout()
    plot_file_name = "plot_sobols_first[{}]".format(output_column)
    plt.savefig(os.path.join(campaign_work_dir, plot_file_name),
                dpi=400)

    ###############################################################
    # yml_results contains all campaign info and analysis results #
    # it will be saved in sobols.yml file                         #
    ###############################################################
    S = ruamel.yaml.scalarstring.DoubleQuotedScalarString
    yml_results = ruamel.yaml.comments.CommentedMap()
    yml_results.update({"campaign_info": {}})
    yml_results["campaign_info"].update({
        "name": S(campaign._active_app_name),
        "work_dir": S(campaign.work_dir),
        "num_runs": campaign.campaign_db.get_num_runs(),
        "output_column": S(output_column),
        "polynomial_order": polynomial_order,
        "sampler": S(SA_campaign_config["sampler_name"]),
        "distribution_type": S(SA_campaign_config["distribution_type"]),
        "sparse": S(SA_campaign_config["sparse"]),
        "growth": S(SA_campaign_config["growth"])
    })
    if sampler_name == "SCSampler":
        yml_results["campaign_info"].update({
            "quadrature_rule": S(SA_campaign_config["quadrature_rule"]),
            "midpoint_level1": S(SA_campaign_config["midpoint_level1"]),
            "dimension_adaptive": S(SA_campaign_config["dimension_adaptive"])
        })
    elif sampler_name == "PCESampler":
        yml_results["campaign_info"].update({
            "rule": S(SA_campaign_config["quadrature_rule"]),
        })

    ROUND_NDIGITS = 4
    for param in SA_campaign_config["selected_vary_parameters"]:
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

    res_file_name = os.path.join(campaign_work_dir, "sobols.yml")
    print(res_file_name)
    with open(res_file_name, "w") as outfile:
        yaml.dump(yml_results, outfile)


def init_SA_campaign(plugin_name, campaign_name, campaign_config,
                     polynomial_order, campaign_work_dir):

    ######################################
    # delete campaign_work_dir is exists #
    ######################################
    if os.path.exists(campaign_work_dir):
        rmtree(campaign_work_dir)
    os.makedirs(campaign_work_dir)

    #####################
    # Create an encoder #
    #####################
    encoder = uq.encoders.GenericEncoder(
        template_fname=os.path.join(get_plugin_path(plugin_name),
                                    "templates",
                                    campaign_config["encoder_template_fname"]
                                    ),
        delimiter=campaign_config["encoder_delimiter"],
        target_filename=campaign_config["encoder_target_filename"]
    )

    # output_filename = campaign_config["decoder_output_column"]
    # output_column = campaign_config["params"]["out_file"]["default"]
    # decoder = uq.decoders.SimpleCSV(
    #     target_filename=output_filename,
    #     output_columns=[output_column]
    # )

    ###########################
    # Set up a fresh campaign #
    ###########################
    db_location = "sqlite:///" + campaign_work_dir + "/campaign.db"

    actions = uq.actions.Actions(
        uq.actions.CreateRunDirectory(root=campaign_work_dir, flatten=True),
        uq.actions.Encode(encoder),
    )

    campaign = uq.Campaign(
        name=campaign_name,
        db_location=db_location,
        work_dir=campaign_work_dir
    )

    ################################
    # Add the SA-Sampler app #
    ################################
    campaign.add_app(
        name=campaign_name,
        params=campaign_config["params"],
        actions=actions
    )

    ######################
    # parameters to vary #
    ######################
    vary = {}
    for param in campaign_config["selected_vary_parameters"]:
        pprint(campaign_config[
            "vary_parameters_range"][param])
        lower_value = campaign_config[
            "vary_parameters_range"][param]["range"][0]
        upper_value = campaign_config[
            "vary_parameters_range"][param]["range"][1]
        if campaign_config["distribution_type"] == "DiscreteUniform":
            vary.update({param: cp.DiscreteUniform(lower_value, upper_value)})
        elif campaign_config["distribution_type"] == "Uniform":
            vary.update({param: cp.Uniform(lower_value, upper_value)})

    ####################
    # create Sampler #
    ####################
    sampler_name = campaign_config["sampler_name"]
    if sampler_name == "SCSampler":
        sampler = uq.sampling.SCSampler(
            vary=vary,
            polynomial_order=polynomial_order,
            quadrature_rule=campaign_config["quadrature_rule"],
            growth=campaign_config["growth"],
            sparse=campaign_config["sparse"],
            midpoint_level1=campaign_config["midpoint_level1"],
            dimension_adaptive=campaign_config["dimension_adaptive"]
        )
    elif sampler_name == "PCESampler":
        sampler = uq.sampling.PCESampler(
            vary=vary,
            polynomial_order=polynomial_order,
            rule=campaign_config["quadrature_rule"],
            sparse=campaign_config["sparse"],
            growth=campaign_config["growth"]
        )
    # TODO: add other sampler here

    ###########################################
    # Associate the sampler with the campaign #
    ###########################################
    campaign.set_sampler(sampler)

    #########################################
    # draw all of the finite set of samples #
    #########################################
    campaign.execute().collate()

    #########################################
    # extract generated runs id by campaign #
    #########################################
    runs_dir = []
    for _, run_info in campaign.campaign_db.runs(
            status=uq.constants.Status.NEW
    ):
        runs_dir.append(run_info["run_name"])

    campaign_dir = campaign.campaign_db.campaign_dir()

    ######################################################
    # backup campaign files, i.e, *.db, *.json, *.pickle #
    ######################################################
    backup_campaign_files(campaign.work_dir)

    print("=" * 50)
    print("With user's specified parameters for {}".format(sampler_name))
    print("campaign name : {}".format(campaign_name))
    print("number of generated runs : {}".format(len(runs_dir)))
    print("campaign dir : {}".format(campaign_work_dir))
    print("=" * 50)

    return runs_dir, campaign_dir


def load_SA_campaign_config(plugin_name, SA_config_file):
    SA_campaign_config = yaml.load(
        open(SA_config_file),
        Loader=yaml.SafeLoader
    )
    #####################################################
    # load parameter space for the easyvvuq sampler app #
    #####################################################
    sampler_params_json_PATH = os.path.join(get_plugin_path(plugin_name),
                                            "templates",
                                            "params.json"
                                            )
    sampler_params = json.load(open(sampler_params_json_PATH))

    #####################################################
    # add loaded campaign params to SA_campaign_config #
    #####################################################
    SA_campaign_config.update({"params": sampler_params})

    return SA_campaign_config


def backup_campaign_files(campaign_work_dir):
    backup_dir = os.path.join(campaign_work_dir, "backup")
    # delete backup folder
    if os.path.exists(backup_dir):
        rmtree(backup_dir)
    os.mkdir(backup_dir)

    local(
        "rsync -av -m -v \
            --include='*.db' \
            --include='*.pickle' \
            --include='*.json' \
            --exclude='*' \
            {}/  {} ".format(campaign_work_dir, backup_dir)
    )


def load_campaign_files(campaign_work_dir):
    backup_dir = os.path.join(campaign_work_dir, "backup")
    local(
        "rsync -av -m -v \
            --include='*.db' \
            --include='*.pickle' \
            --include='*.json' \
            --exclude='*' \
            {}/  {} ".format(backup_dir, campaign_work_dir)
    )
