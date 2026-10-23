# -*- coding: utf-8 -*-
#
# This source file is part of the FabSim software toolkit,
# which is 151distributed under the BSD 3-Clause license.
# Please refer to LICENSE for detailed information regarding the licensing.
#
# This file contains FabSim definitions specific to fabFlee.

try:
    from fabsim.base.fab import *
except ImportError:
    from base.fab import *

# Import V&V primitives.
try:
    import fabsim.VVP.vvp as vvp
except ImportError:
    import VVP.vvp as vvp

import glob
import csv
import os
import re
import numpy as np
import pandas as pd
from shutil import copyfile, rmtree, move

# Add local script, blackbox and template path.
add_local_paths("FabFlee")

@task
@load_plugin_env_vars("FabFlee")
# Syntax: fabsim localhost get_flee_location
def get_flee_location():
    """
    Print the $flee_location env variable for the target machine.
    """
    update_environment()
    print(env.machine_name, env.flee_location)


@task
@load_plugin_env_vars("FabFlee")
# Syntax: fabsim localhost sflee:burundi,simulation_period=10
def sflee(config, simulation_period, **args):
    """ Submit a Flee job to the remote queue.
    The job results will be stored with a name pattern as
    defined in the environment,
    e.g. car-abcd1234-localhost-4
    config :
        config directory to use for the simulation script, e.g. config=car2014
    simulation_period : length of the simulation in days.
    Keyword arguments:
            cores : number of compute cores to request
            wall_time : wall-time job limit
            memory : memory per node
    """
    update_environment(args, {"simulation_period": simulation_period})
    with_config(config)
    execute(put_configs, config)
    job(dict(script='flee', cores=1, wall_time='0:15:0', memory='2G'), args)



@task
@load_plugin_env_vars("FabFlee")
# Syntax: fabsim localhost sync_flee
def sync_flee():
    """
    Synchronize the Flee version, so that the remote machine has the latest
    version from localhost.
    """
    update_environment()
    flee_location_local = user_config["localhost"].get(
        "flee_location", user_config["default"].get("flee_location"))

    rsync_project(
        local_dir=flee_location_local + '/',
        remote_dir=env.flee_location
    )


@task
# Syntax: fabsim localhost flees:<config>,simulation_period=<value>,<optional_args>
def flees(config, simulation_period, **args):
    """
      Save relevant arguments to a Python or numpy list.
    """
    print(args)

    # Generate config directories, copying from the config provided,
    # and adding a different generated test.csv in each directory.
    # Run the flee() a number of times.


@task
@load_plugin_env_vars("FabFlee")
# Syntax: fabsim localhost flee_ensemble:<config>,simulation_period=<value>,label=<label>,<optional_args>
# label=<label> and <optional_args> are optional and command can run without them
def flee_ensemble(config, simulation_period, script='flee', label="", **args):
    """
    Submits an ensemble of dummy jobs.
    One job is run for each file in <config_file_directory>/flee_test/SWEEP.
    """
    update_environment(args)
    with_config(config)
    path_to_config = find_config_file_path(config)
    print("local config file path at: %s" % path_to_config)
    sweep_dir = path_to_config + "/SWEEP"
    env.script = script
    env.input_name_in_config = 'flee.txt'
    env.simulation_period = simulation_period

    if hasattr(env, 'NoEnvScript'):
        del env['NoEnvScript']

    # Re-add support for labels, which are overwritten by runensemble.
    if len(label) > 0:
        print("adding label: ", label)
        env.job_name_template += "_{}".format(label)

    run_ensemble(config, sweep_dir, **args)


def _generate_flee_weather_sweep(sweep_dir, mode, seeds, configs, q1s, q2s, Qs):
    """
    (Re)generate <config>/SWEEP/<label>/wc_args for a weather-coupled ensemble.

    Each run gets a tiny wc_args file holding its own weather flags, which the
    flee_weather template reads. Labels follow the original single-scale scripts'
    output-folder names (e.g. M1_seed3 and q10.15_q20.75_Q8000_seed1).
    """
    flags_for = {
        "M0": "--weather_coupling=false",
        "M1": "--weather_coupling=true --use_precipitation=true --use_discharge=false",
        "M2": "--weather_coupling=true --use_precipitation=false --use_discharge=true",
        "M3": "--weather_coupling=true --use_precipitation=true --use_discharge=true",
    }

    if os.path.exists(sweep_dir):
        rmtree(sweep_dir)
    os.makedirs(sweep_dir)

    def _add(item_label, wc_args):
        run_dir = os.path.join(sweep_dir, item_label)
        os.makedirs(run_dir)
        with open(os.path.join(run_dir, "wc_args"), "w") as handle:
            handle.write(wc_args.strip() + "\n")

    labels = []
    if mode == "core":
        for cfg in configs.split():
            for seed in seeds.split():
                label = "%s_seed%s" % (cfg, seed)
                _add(label, "%s --seed=%s" % (flags_for[cfg], seed))
                labels.append(label)
    elif mode == "sweep":
        for q1 in q1s.split():
            for q2 in q2s.split():
                for Q in Qs.split():
                    for seed in seeds.split():
                        label = "q1%s_q2%s_Q%s_seed%s" % (q1, q2, Q, seed)
                        _add(label,
                             "--weather_coupling=true --use_precipitation=true "
                             "--use_discharge=true --q1=%s --q2=%s --Q=%s --seed=%s"
                             % (q1, q2, Q, seed))
                        labels.append(label)
    else:
        raise RuntimeError(
            "[ERROR] mode must be 'core' or 'sweep', got: %s" % mode)
    return labels


@task
@load_plugin_env_vars("FabFlee")
# Syntax: fabsim archer2 flee_weather_ensemble:ssudan2016_weather,mode=sweep,simulation_period=426
def flee_weather_ensemble(config, simulation_period="426", mode="sweep",
                          seeds="1 2 3 4 5",
                          configs="M0 M1 M2 M3",
                          q1s="0.10 0.15 0.20",
                          q2s="0.60 0.75 0.90",
                          Qs="6000 8000 10000",
                          label="", pj_type="", simsetting="", **args):
    """
    Submit a weather-coupled single-scale Flee ensemble (M0-M3 counterfactuals
    and/or the q1/q2/Q threshold sensitivity sweep).

    This drives the single-scale weather_coupling patch: each run gets its own
    flags via SWEEP/<label>/wc_args, read by templates/flee_weather.

    mode:
        'core'  -> configs x seeds            (default 4 x 5 = 20 runs)
        'sweep' -> q1s x q2s x Qs x seeds     (default 3 x 3 x 3 x 5 = 135 runs)

    simsetting: settings file inside the config to use. Empty => chosen by mode:
    'simsetting_core.yml' for core (agent/link logs ON, matching the paper's
    core setup) and 'simsetting.yml' for the sweep (logs off, saves disk).

    pj_type: optional ensemble backend chosen on the command line, e.g.
    pj_type=qcg (QCG-PilotJob), pj_type=rp (RADICAL-Pilot), or
    pj_type=slurm-array / slurm-manager. Left empty => traditional submission
    (one SLURM job per run). This stays a CLI choice until the ensemble backend
    for real apps is standardised upstream.

    The re-generated SWEEP/<label>/ dirs keep the original single-scale scripts'
    output-folder naming (e.g. M1_seed3 and q10.15_q20.75_Q8000_seed1).
    """
    update_environment(args)
    with_config(config)
    path_to_config = find_config_file_path(config)
    print("local config file path at: %s" % path_to_config)
    sweep_dir = os.path.join(path_to_config, "SWEEP")

    labels = _generate_flee_weather_sweep(
        sweep_dir, mode, seeds, configs, q1s, q2s, Qs)
    print("[INFO] generated %d weather sweep runs in %s"
          % (len(labels), sweep_dir))

    env.script = "flee_weather"
    env.simulation_period = simulation_period

    # Pick the settings file for this ensemble: the core setup has agent/link
    # logs ON (simsetting_core.yml), the sweep keeps them off to save disk.
    if not simsetting:
        simsetting = "simsetting_core.yml" if mode == "core" else "simsetting.yml"
    env.simsetting_file = simsetting
    print("[INFO] simsetting: %s" % simsetting)

    # pj_type is an explicit command-line choice (e.g. pj_type=qcg) rather than
    # a hardcoded default, until the team standardises the ensemble backend for
    # real apps. Empty => traditional submission (one SLURM job per run).
    if pj_type:
        env.pj_type = pj_type
        # PilotJob backends (QCG/RP) launch each task directly inside the
        # existing allocation, so the machine's default run_command
        # (archer2: "srun ...") must not be nested inside the task script --
        # nesting fails with a cpu-bind error. Non-pilot runs keep it.
        if pj_type.lower() in ("qcg", "rp"):
            env.run_command = ""

    if hasattr(env, 'NoEnvScript'):
        del env['NoEnvScript']

    if len(label) > 0:
        print("adding label: ", label)
        env.job_name_template += "_{}".format(label)

    run_ensemble(config, sweep_dir, **args)


# One-at-a-time (OAT) epistemic/structural UQ axes. Each axis sweeps a single
# move_rules setting that lives only in simsetting.yml (SimulationSettings), not
# in run_par.py command-line flags. Only the NEW levels are run; the reference
# level (awareness 3, speed 360) is the existing core ensemble and is reused.
_EPISTEMIC_AXES = {
    "awareness": ("awareness_level", ["1", "2"], "aw"),
    "speed": ("max_move_speed", ["200", "280"], "spd"),
}


def _write_epistemic_simsetting(base_path, out_path, key, value):
    """
    Copy the config's base simsetting.yml to out_path, overriding one move_rules
    knob and switching agent/link logging off (this analysis needs only out.csv:
    ARD from 'Total error', R from the per-camp 'sim' columns).
    """
    with open(base_path) as handle:
        text = handle.read()
    text = re.sub(r"(?m)^(\s*)%s:\s*\S+.*$" % key,
                  r"\g<1>%s: %s" % (key, value), text)
    text = re.sub(r"(?m)^(\s*)agent:\s*\S+.*$", r"\g<1>agent: 0", text)
    text = re.sub(r"(?m)^(\s*)link:\s*\S+.*$", r"\g<1>link: 0", text)
    with open(out_path, "w") as handle:
        handle.write(text)


def _generate_epistemic_sweep(sweep_dir, axis, configs, seeds):
    """
    (Re)generate <config>/SWEEP/<label>/ for one OAT epistemic axis.

    Each run dir gets a per-run simsetting.yml (one structural knob overridden,
    logging off) plus a wc_args holding its M0-M3 flags and seed. Climate stays
    at the reference thresholds (no --q1/--q2/--Q => runner defaults), so the
    only free parameters are the swept knob and the seed.
    """
    if axis not in _EPISTEMIC_AXES:
        raise RuntimeError("[ERROR] axis must be one of %s, got: %s"
                           % (sorted(_EPISTEMIC_AXES), axis))
    key, levels, prefix = _EPISTEMIC_AXES[axis]

    flags_for = {
        "M0": "--weather_coupling=false",
        "M1": "--weather_coupling=true --use_precipitation=true --use_discharge=false",
        "M2": "--weather_coupling=true --use_precipitation=false --use_discharge=true",
        "M3": "--weather_coupling=true --use_precipitation=true --use_discharge=true",
    }

    base_settings = os.path.join(os.path.dirname(sweep_dir), "simsetting.yml")
    if not os.path.exists(base_settings):
        raise RuntimeError("[ERROR] base settings not found: %s" % base_settings)

    if os.path.exists(sweep_dir):
        rmtree(sweep_dir)
    os.makedirs(sweep_dir)

    labels = []
    for level in levels:
        for cfg in configs.split():
            for seed in seeds.split():
                label = "%s%s_%s_seed%s" % (prefix, level, cfg, seed)
                run_dir = os.path.join(sweep_dir, label)
                os.makedirs(run_dir)
                with open(os.path.join(run_dir, "wc_args"), "w") as handle:
                    handle.write("%s --seed=%s\n" % (flags_for[cfg], seed))
                _write_epistemic_simsetting(
                    base_settings, os.path.join(run_dir, "simsetting.yml"),
                    key, level)
                labels.append(label)
    return labels


@task
@load_plugin_env_vars("FabFlee")
# Syntax: fabsim archer2 flee_epistemic_ensemble:ssudan2016_weather,axis=awareness,pj_type=qcg
def flee_epistemic_ensemble(config, simulation_period="426", axis="awareness",
                            seeds="1 2 3 4 5", configs="M0 M1 M2 M3",
                            label="", pj_type="", **args):
    """
    Submit one axis of the OAT epistemic/structural UQ sweep.

    Varies a single structural parameter across the NEW levels only (the
    reference cell -- awareness 3 / speed 360 -- is the existing core ensemble
    and is NOT re-run):

        axis='awareness' -> awareness_level in {1, 2}
        axis='speed'     -> max_move_speed in {200, 280} km/day

    Climate is held at the reference thresholds (runner defaults q1=0.15,
    q2=0.75, Q=8000) and every M0-M3 configuration runs all 5 seeds, so each
    axis is 2 x 4 x 5 = 40 runs.

    The swept knob is a SimulationSettings field read from simsetting.yml, not a
    run_par.py flag, so each run ships its own SWEEP/<label>/simsetting.yml
    (based on the config's simsetting.yml, logging off), which run_ensemble
    copies over the config copy in the job directory. Per-run M0-M3 flags and
    the seed go in SWEEP/<label>/wc_args as usual.

    Give each concurrent axis a distinct job_desc (e.g. job_desc=_awareness) so
    the two jobs stage into separate remote config dirs and cannot race.
    """
    update_environment(args)
    with_config(config)
    path_to_config = find_config_file_path(config)
    print("local config file path at: %s" % path_to_config)
    sweep_dir = os.path.join(path_to_config, "SWEEP")

    labels = _generate_epistemic_sweep(sweep_dir, axis, configs, seeds)
    print("[INFO] generated %d epistemic (%s) runs in %s"
          % (len(labels), axis, sweep_dir))

    env.script = "flee_weather"
    env.simulation_period = simulation_period
    # Each run ships its own simsetting.yml (logging off) in SWEEP/<label>/.
    env.simsetting_file = "simsetting.yml"

    if pj_type:
        env.pj_type = pj_type
        # PilotJob backends (QCG/RP) launch each task inside the existing
        # allocation, so the machine run_command ("srun ...") must not be nested.
        if pj_type.lower() in ("qcg", "rp"):
            env.run_command = ""

    if hasattr(env, 'NoEnvScript'):
        del env['NoEnvScript']

    if len(label) > 0:
        print("adding label: ", label)
        env.job_name_template += "_{}".format(label)

    run_ensemble(config, sweep_dir, **args)


def load_module_from_path(moduleName, PATH_to_module):
    import importlib

    try:
        importlib.import_module(moduleName)
        print("module {} is already in the PYTHONPATH "
              "and correctly loaded ".format(moduleName)
              )
    except ModuleNotFoundError:
        # add module PATH to PYTHONPATH
        sys.path.insert(0, PATH_to_module)
        try:
            # check again to see if the input moduleName can be loaded from
            # PATH_to_module or not
            importlib.import_module(moduleName)
            print("module {} loaded correctly form {} ".format(
                moduleName, env.flare_location)
            )
        except ModuleNotFoundError:
            raise ValueError(
                "The input PATH = {} for {} is not VALID!".format(
                    env.flare_location, moduleName)
            )
            sys.exit()
    except Exception as exception:
        print('Error: ', exception)
        sys.exit()


@task
@load_plugin_env_vars("FabFlee")
# Syntax: fabsim localhost flare_local:<config>,simulation_period=<value>,out_dir=<output_directory>,file_suffix=<suffix>
def flare_local(config, simulation_period, out_dir="", file_suffix=""):
    """
    Run an instance of Flare on the local host.
    """

    load_module_from_path(
        moduleName="flare", PATH_to_module=env.flare_location
    )
    load_module_from_path(
        moduleName="flee", PATH_to_module=env.flee_location
    )

    if len(out_dir) == 0:
        out_dir = "{}_single".format(config)

    flare_out_dir = "{}/results-flare/{}".format(
        get_plugin_path("FabFlee"), out_dir
    )
    config_dir = "{}/config_files/{}".format(
        get_plugin_path("FabFlee"), config
    )

    local("mkdir -p {}/input_csv".format(flare_out_dir))

    # load run_flare function from script directory
    from .scripts.run_flare import run_flare

    run_flare(
        config_dir="{}/input_csv".format(config_dir),
        flare_out_dir="{}/input_csv/conflicts{}.csv".format(
            flare_out_dir, file_suffix),
        simulation_period=int(simulation_period),
        file_suffix=file_suffix
    )


@task
@load_plugin_env_vars("FabFlee")
# Syntax: fabsim localhost flare_ensemble:<config>,simulation_period=<value>,N=<instances>,out_dir=<output_directory>,file_suffix=<suffix>
def flare_ensemble(config, simulation_period, N, out_dir, file_suffix=""):
    """
    Run an ensemble of flare instances locally.
    config: configuration directory.
    simulation_period: simulation period in days.
    N: number of instances in ensemble.
    out_dir: base output subdirectory in flare-results.

    Parameters:
    - config : str
        Configuration directory to use for the simulation.
    - simulation_period : int
        Length of the simulation in days.
    - out_dir : str, optional
        Path to the output directory where results will be saved (default: "").
    - file_suffix : str, optional
        Custom suffix to append to output files (default: "").

    Example:
        flare_local(config="config_name", simulation_period=10, 
                    out_dir="results/", file_suffix="_test")
    """
    for i in range(0, int(N)):
        instance_out_dir = "%s/%s" % (out_dir, i)
        flare_local(config, simulation_period,
                    instance_out_dir, file_suffix=file_suffix)


@task
@load_plugin_env_vars("FabFlee")
# Syntax: fabsim localhost couple_flare_to_flee:<config>,flare_out=<flare_output_directory>
def couple_flare_to_flee(config, flare_out="flare-out-scratch"):
    """
    Convert Flare output and prepare it for a Flee ensemble run.

    Parameters:
    - config : str
        Configuration directory to use for the conversion process.
    - flare_out : str, optional
        Directory containing the Flare output files (default: "flare-out-scratch").

    Example:
        couple_flare_to_flee(config="config_name",flare_out="flare-results/output")
    """

    with_config(config)
    config_dir = env.job_config_path_local
    local("rm -rf %s/SWEEP" % (config_dir))
    local("mkdir -p %s/SWEEP" % (config_dir))
    local("cp -r %s/results-flare/%s/* %s/SWEEP/"
          % (get_plugin_path("FabFlee"), flare_out, config_dir))


@task
@load_plugin_env_vars("FabFlee")
# Syntax: fabsim localhost flee_conflict_forecast:<config>,simulation_period=<value>,N=<instances>,<optional_args>
def flee_conflict_forecast(config, simulation_period, N, **args):
    """
    Run a Flare ensemble, convert the output to Flee ensemble input, 
    and execute the Flee ensemble.

    This method performs the following steps:
    1. Runs a Flare ensemble simulation.
    2. Converts the Flare output into Flee input format.
    3. Runs a Flee ensemble simulation based on the converted input.
    4. Optionally visualizes the Flee output with uncertainty metrics.

    Parameters:
    - config : str
        Configuration directory to use for the simulations.
    - simulation_period : int
        Length of the simulation in days.
    - N : int
        Number of ensemble instances to run for Flare and Flee.
    - **args : dict
        Additional keyword arguments for simulation configuration.

    Example:
        flee_conflict_forecast(config="config_name",simulation_period=30,N=10,cores=4,wall_time="02:00:00")
    """

    update_environment(args)

    local("rm -rf %s/results-flare/flare-out-scratch/*" %
          (get_plugin_path("FabFlee")))
    flare_ensemble(config, simulation_period, N, "flare-out-scratch")

    couple_flare_to_flee(config, flare_out="flare-out-scratch")

    # config_dir = "%s/config_files/%s" % (get_plugin_path("FabFlee"), config)
    # local("mkdir -p %s/SWEEP" % (config_dir))
    # local("cp -r %s/results-flare/flare-out-scratch/* %s/SWEEP/"
    # % (get_plugin_path("FabFlee"), config_dir))

    flee_ensemble(config, simulation_period, **args)


# Flee parallelisation tasks
@task
@load_plugin_env_vars("FabFlee")
# Syntax: fabsim localhost pflee:<config>,simulation_period=<value>,profile=<True/False>,<optional_args>
def pflee(config, simulation_period, profile=False, **args):
    """ Submit a Pflee job to the remote queue.
    The job results will be stored with a name pattern as defined
    in the environment, e.g. car-abcd1234-localhost-4
    config :
        config directory to use for the simulation script, e.g. config=car2014
    Keyword arguments:
            cores : number of compute cores to request
            wall_time : wall-time job limit
            memory : memory per node
    """
    '''
    update_environment({"input_directory": "%s/config_files/%s/input_csv"
                        % (get_plugin_path("FabFlee"), config),
                        "validation_data_directory":
                        "%s/config_files/%s/source_data"
                        % (get_plugin_path("FabFlee"), config)})
    print_local_environment()
    '''
    update_environment(args, {"simulation_period": simulation_period})
    with_config(config)
    execute(put_configs, config)
    if bool(profile) is True:
        job(dict(script='pflee_profile', wall_time='0:30:0', memory='2G'), args)
    else:
        job(dict(script='pflee', wall_time='0:30:0', memory='2G'), args)


@task
@load_plugin_env_vars("FabFlee")
# Syntax: fabsim localhost pflee_test:<config>
def pflee_test(config, pmode="advanced", N="100000", **args):
    """
    Run a short parallel test with a particular config.
    """
    update_environment(args, {"simulation_period": 100,
                              "flee_parallel_mode": pmode,
                              "flee_num_agents": int(N)
                              }
                       )
    with_config(config)
    execute(put_configs, config)
    job(dict(script='pflee_test', wall_time='0:15:0', memory='2G'), args)


@task
@load_plugin_env_vars("FabFlee")
# Syntax: fabsim localhost pflee_pmode_compare:<config>
def pflee_pmode_compare(config, cores, N="100000", **args):
    """
    Run a short parallel test with a particular config. 60 min limit per run.
    """
    # maps to args in test_par.py
    for pmode in ["advanced", "classic", "adv-lolat", "cl-hilat"]:
        update_environment(args, {"simulation_period": 10,
                                  "flee_parallel_mode": pmode,
                                  "flee_num_agents": int(N)
                                  }
                           )
        with_config(config)
        execute(put_configs, config)
        job(dict(script='pflee_test', wall_time='1:00:0',
                 memory='2G', cores=cores, label=pmode), args)


# Syntax: fabsim localhost pflee_report:<results_key>
def pflee_report(results_key):
    """
    Generate a performance report for Pflee results.
    """
    for item in glob.glob("{}/*{}*/perf.log".format(env.local_results,
                                                    results_key)):
        print(item)
        with open(item) as csvfile:
            perf = csv.reader(csvfile)
            for k, e in enumerate(perf):
                if k == 1:
                    print(float(e[1]))

    # local("grep main {}/{}/perf.log".format(env.local_results,results_key))


@task
@load_plugin_env_vars("FabFlee")
def pflee_ensemble(config, simulation_period, **args):
    flee_ensemble(config, simulation_period, script='pflee', **args)


# Coupling Flee and food security tasks
@task
@load_plugin_env_vars("FabFlee")
# Syntax: fabsim localhost food_flee:<config>,simulation_period=<value>
def food_flee(config, simulation_period, **args):
    """ Submit a Flee job to the remote queue.
    The job results will be stored with a name pattern as defined
    in the environment, e.g. car-abcd1234-localhost-4
    config :
        config directory to use for the simulation script, e.g. config=car2014
    Keyword arguments:
            cores : number of compute cores to request
            wall_time : wall-time job limit
            memory : memory per node
    """
    update_environment({"input_directory": "%s/config_files/%s/input_csv"
                        % (get_plugin_path("FabFlee"), config),
                        "validation_data_directory":
                        "%s/config_files/%s/source_data"
                        % (get_plugin_path("FabFlee"), config)})
    # print_local_environment()
    update_environment(args, {"simulation_period": simulation_period})
    with_config(config)
    execute(put_configs, config)
    job(dict(script='flee_food', wall_time='0:15:0', memory='2G'), args)


@task
@load_plugin_env_vars("FabFlee")
# Syntax: fabsim localhost compare_food:food_flee_conflict_name_localhost_16
def compare_food(output_dir_1=""):
    """
    Compare results of the food based simulation with the original
    flee results throughout the whole simulation.
    Syntax:
        fabsim localhost compare_food:food_flee_conflict_name_localhost_16
        **or any name the food directory you want to use has.
        Make sure that the non-food one exists as well.
    """
    local("mkdir -p %s/%s/comparison" % (env.results_path, output_dir_1))
    output_dir_2 = output_dir_1.partition("_")[2]
    local("python3 %s/compare.py %s/%s %s/%s"
          % (env.flee_location,
             env.results_path, output_dir_1,
             env.results_path, output_dir_2))


@task
@load_plugin_env_vars("FabFlee")
# Syntax: fabsim localhost plot_flee_profile:output_dir,profiler
def plot_flee_profile(output_dir="", profiler="gprof2dot"):
    """
    Plot a Flee profile using the specified profiler.

    This method generates a visual representation of a Flee profile, utilizing 
    either `gprof2dot` or `snakeviz` as the profiler tool. It requires additional 
    dependencies such as Graphviz for rendering the output and optionally `eog` 
    for viewing the generated image.

    """

    if profiler == "gprof2dot":
        # Use gprof2dot to generate the profile visualization
        local(f"gprof2dot --colour-nodes-by-selftime -f pstats {env.local_results}/{output_dir}/prof.log | dot -Tpng -o {env.local_results}/{output_dir}/profile.png")
        
        # Open the generated profile image with eog
        local(f"eog {env.local_results}/{output_dir}/profile.png")
    elif profiler == "snakeviz":
        # Use snakeviz to visualize the profile
        local(f"snakeviz {env.local_results}/{output_dir}/prof.log")
        print(f"Snakeviz output saved as HTML: {env.local_results}/{output_dir}/profile.html")
    else:
        print("Invalid profiler choice. Please choose 'gprof2dot' or 'snakeviz'.")


# Post-processing tasks
@task
@load_plugin_env_vars("FabFlee")
# Syntax: fabsim localhost plot_output:flee_conflict_name_localhost_16(,graphs_dir_name)
def plot_output(output_dir="", graphs_dir=""):
    """ 
    Plot generated output results using plot-flee-output.py. 
    """
    local("mkdir -p %s/%s/%s" % (env.local_results, output_dir, graphs_dir))

    # import plot_flee_output.py from env.flee_location
    # when we have pip flee installation option, this part should be changed
    for p in env.flee_location.split(":"):
        sys.path.insert(0, p)

    from flee.postprocessing.plot_flee_output import plot_flee_output
    plot_flee_output(
        os.path.join(env.local_results, output_dir),
        os.path.join(env.local_results, output_dir, graphs_dir)
    )
    '''
    local("python3 %s/plot-flee-output.py %s/%s %s/%s/%s"
          % (env.flee_location,
             env.local_results, output_dir,
             env.local_results, output_dir, graphs_dir))
    '''


@task
@load_plugin_env_vars("FabFlee")
def create_agents_video(output_dir=""):
    """
    Generate PNGs and videos for links based on simulation outputs.
    """
    import sys
    from flee.postprocessing.video_agents import process_files

    try:
        # Create directories if they do not exist
        local("mkdir -p %s/%s" % (env.local_results, output_dir))

        # Dynamically add flee.postprocessing path for imports
        for p in env.flee_location.split(":"):
            sys.path.insert(0, p)

        # Absolute path to the output directory
        full_output_dir = os.path.join(env.local_results, output_dir)

        print(f"Processing agents PNGs and videos in directory: {full_output_dir}")
        
        # Call the PNG and video generation function from the script
        process_files(full_output_dir)
        
        print(f"Agents PNGs and video successfully generated in: {full_output_dir}")
    except Exception as e:
        print(f"Error occurred in process_agents_output: {traceback.format_exc()}")


@task
@load_plugin_env_vars("FabFlee")
def create_links_video(output_dir=""):
    """
    Generate PNGs and videos for links based on simulation outputs.
    """
    import sys
    from flee.postprocessing.video_links import process_files

    try:
        # Create directories if they do not exist
        local("mkdir -p %s/%s" % (env.local_results, output_dir))

        # Dynamically add flee.postprocessing path for imports
        for p in env.flee_location.split(":"):
            sys.path.insert(0, p)

        # Absolute path to the output directory
        full_output_dir = os.path.join(env.local_results, output_dir)

        print(f"Processing links PNGs and videos in directory: {full_output_dir}")
        
        # Call the PNG and video generation function from the script
        process_files(full_output_dir)
        
        print(f"Links PNGs and video successfully generated in: {full_output_dir}")
    except Exception as e:
        print(f"Error occurred in process_links_output: {traceback.format_exc()}")


@task
@load_plugin_env_vars("FabFlee")
# Syntax: fabsim localhost flee_compare:<model#1>,<model#2>,...,,model#n>
def flee_compare(*models,output_dir=""):
    """
    Compare results of flee simulations for a conflict scenario.
    """
    output_dir = models[0].partition("_")[0]

    local("mkdir -p %s/%s_comparison" % (env.local_results, output_dir))

    output_dir = "%s/%s_comparison" % (env.local_results, output_dir)

    from flee.postprocessing.plot_flee_compare import plot_flee_compare
    plot_flee_compare(*models,data_dir=env.local_results,output_dir=output_dir)
    
    
@task
@load_plugin_env_vars("FabFlee")
# Syntax: fabsim localhost plot_forecast:flee_conflict_name_localhost_16(,graphs_dir_name)
def plot_forecast(output_dir="", region_names=""):
    """ 
    Plot generated output results using plot-flee-forecast.py. 
    """
    # import plot_flee_forecast.py from env.flee_location
    # when we have pip flee installation option, this part should be changed
    for p in env.flee_location.split(":"):
        sys.path.insert(0, p)
    from flee.postprocessing.plot_flee_forecast import plot_flee_forecast

    region_names = []
    if len(region_names) > 0:
        region_names = region_names.split(';')

    input_dir = os.path.join(env.local_results, output_dir)
    if len(region_names) == 0:
        # find all region names
        data_dir = os.path.join(input_dir, "RUNS")
        dir_names = os.listdir(data_dir)
        for dir_names in os.listdir(data_dir):
            region_name = dir_names.rsplit('_', 1)[0]
            if region_name not in region_names:
                region_names.append(region_name)
        region_names.sort()

    plot_flee_forecast(
        input_dir=input_dir,
        region_names=region_names
    )


@task
@load_plugin_env_vars("FabFlee")
# Syntax: fabsim localhost plot_flee_profile:output_dir=<output_directory>
def cflee(config, coupling_type="file", weather_coupling="False",
          num_instances="1", instance_cores="1",
          job_wall_time="00:12:00", ** args):
    """ Submit a cflee (coupling flee) job to the remote queue.
    The job results will be stored with a name pattern as defined
    Required Keyword arguments:
        config :
            config directory to use for the simulation script,
            e.g. config=mscalecity
        coupling_type :
            the coupling model, currently two models are implemented :
            (1) file couping, and (2) muscle3
            acceptable input set : file / muscle3
    Example:
        fabsim eagle_hidalgo cflee:ssudan-mscale-test,coupling_type=file,
        weather_coupling=False,num_instances=2,instance_cores=2,TestOnly=True

        fabsim eagle_hidalgo cflee:ssudan-mscale-test,coupling_type=muscle3,
        weather_coupling=False,num_instances=2,instance_cores=2,TestOnly=True

        fabsim eagle_hidalgo cflee:ssudan-mscale-test,coupling_type=file,
        weather_coupling=True,num_instances=10,instance_cores=4

    """
    update_environment(args, {"coupling_type": coupling_type.lower(),
                              "weather_coupling": weather_coupling.lower(),
                              "num_instances": num_instances,
                              "instance_cores": instance_cores,
                              "job_wall_time": job_wall_time
                              }
                       )

    env.cores = int(num_instances) * int(instance_cores) * 2
    env.py_pkg = ["qcg-pilotjob", "pandas", "seaborn", "matplotlib", "jinja2"]
    if coupling_type == "file":
        script = "flee_file_coupling"
    elif coupling_type == "muscle3":
        env.cores += 2
        script = "flee_muscle3_coupling"
        env.py_pkg.append("muscle3")

    label = "coupling_{}_weather_{}".format(
        coupling_type, weather_coupling.lower()
    )
    with_config(config)
    execute(put_configs, config)

    job(dict(script=script, memory="24G", label=label), args)


@task
@load_plugin_env_vars("FabFlee")
# Syntax: fabsim localhost cflee_ensemble:<config>
def cflee_ensemble(config, coupling_type="file", weather_coupling="False",
                   num_workers="1", worker_cores="1",
                   N="1", simulation_period="425",
                   job_wall_time="00:12:00", **args):
    """
    Submit a cflee ensemble job to the remote queue.
    update_environment(args, {"coupling_type": coupling_type,
                              "weather_coupling": weather_coupling.lower(),
                              "num_workers": num_workers,
                              "worker_cores": worker_cores,
                              "job_wall_time": job_wall_time,
                              "simulation_period": simulation_period
                              }
                       )
    env.cores = int(num_workers) * int(worker_cores) * 2
    """

    if coupling_type == 'file':
        script = 'flee_file_coupling'
        label = 'file_coupling'
    elif coupling_type == 'muscle3':
        script = 'flee_muscle3_coupling'
        label = 'muscle3_coupling'
    with_config(config)

    # clean config SWEEP dir if exists
    config_sweep_dir = env.job_config_path_local + "/SWEEP"
    if os.path.exists(config_sweep_dir):
        rmtree(config_sweep_dir)

    # clean flare SWEEP dir if exists
    flare_sweep_dir = "%s/results-flare/%s" % (
        get_plugin_path("FabFlee"), "SWEEP")
    if os.path.exists(flare_sweep_dir):
        rmtree(flare_sweep_dir)

    # run flare
    for file_suffix in ['-0', '-1']:
        flare_ensemble(config, simulation_period=simulation_period,
                       N=N, out_dir="SWEEP", file_suffix=file_suffix)

    # move flare SWEEP dir to config folder
    move(flare_sweep_dir, config_sweep_dir)

    execute(put_configs, config)

    # submit ensambe jobs
    path_to_config = find_config_file_path(config)
    sweep_dir = path_to_config + "/SWEEP"
    env.script = script
    env.label = label
    run_ensemble(config, sweep_dir, **args)


@task
@load_plugin_env_vars("FabFlee")
# Syntax: fabsim localhost flee_and_plot:<config>,simulation_period=10
def flee_and_plot(config, simulation_period, **args):
    """
    Runs Flee and plots the output in a graph subdir
    """
    # update_environment(args, {"simulation_period": simulation_period})
    env.simulation_settings = "simsetting.csv"
    flee(config, simulation_period, **args)
    plot_output("%s" % (env.job_name), "graph")


@task
@load_plugin_env_vars("FabFlee")
# Syntax: fabsim localhost plot_uq_output:flee_conflict_name_localhost_16(,graphs_dir_name)
def plot_uq_output(output_dir="", graphs_dir=""):
    """ 
    Plot generated output results using plot-flee-output.py. 
    """
    local("mkdir -p %s/%s/%s" % (env.local_results, output_dir, graphs_dir))

    # import plot_flee_uq_output.py from env.flee_location
    # when we have pip flee installation option, this part should be changed
    for p in env.flee_location.split(":"):
        sys.path.insert(0, p)

    from flee.postprocessing.plot_flee_uq_output import plot_flee_uq_output
    plot_flee_uq_output(
        os.path.join(env.local_results, output_dir),
        os.path.join(env.local_results, output_dir, graphs_dir)
    )
    '''
    local("python3 %s/plot-flee-uq-output.py %s/%s %s/%s/%s"
          % (env.flee_location,
             env.local_results, output_dir,
             env.local_results, output_dir, graphs_dir))
    '''

# Validation tasks


def vvp_validate_results(output_dir="", **kwargs):
    """ Extract validation results (no dependencies on FabSim env). """

    flee_location_local = user_config["localhost"].get(
        "flee_location", user_config["default"].get("flee_location"))

    local("export PYTHONPATH=%s:${PYTHONPATH}; export FLEE_TYPE_CHECK=False; python3 %s/flee/postprocessing/extract-validation-results.py %s "
          "> %s/validation_results.yml"
          % (flee_location_local, flee_location_local, output_dir, output_dir))

    with open("{}/validation_results.yml".format(output_dir), 'r') as val_yaml:
        validation_results = yaml.load(val_yaml, Loader=yaml.SafeLoader)

        # TODO: make a proper validation metric using a validation schema.
        # print(validation_results["totals"]["Error (rescaled)"])
        #print("Validation {}: {}".format(output_dir.split("/")[-1],
        #                                 validation_results["totals"][
        #                                 "Error (rescaled)"]))
        label = output_dir.split("/")[-1].split("_")[0]
        return [label,validation_results["totals"][f"Error ({env.flee_validation_mode})"]]

    print("Error: vvp_validate_results failed on {}".format(output_dir))
    return None


@task
@load_plugin_env_vars("FabFlee")
# Syntax: fabsim localhost flee_MOO:moo_f1_c1_t3
def flee_MOO(config, simulation_period=60, cores=1, **args):
    """
    Submit a Multi-Objective Optimisation (MOO) Flee job to the remote queue.
    """
    if not isinstance(cores, int):
        cores = int(cores)

    update_environment(
        args,
        {"cores": cores, "simulation_period": simulation_period}
    )

    if cores > 1:
        env.flee_mode = "parallel"
    else:
        env.flee_mode = "serial"
    # set env flag to clear the previous execution folder in case of exists
    env.prevent_results_overwrite = "delete"
    with_config(config)

    ###########################################################################
    # MOO_setting.yaml contains the required setting for executing MOO code,  #
    # so, to be available on the remote machine, we temporally copy           #
    # MOO_setting.yaml file to the target config folder in                    #
    # FabFLee/config_files directory.                                         #
    # later, after execute(put_configs,..), we delete it from config folder   #
    # --------------                                                          #
    # Note :                                                                  #
    #       Hamid: I think this is better solution instead of opening another #
    #       ssh connection to remote machine for transferring only            #
    #       a single file                                                     #
    ###########################################################################
    copyfile(
        src=os.path.join(get_plugin_path("FabFlee"), "MOO_setting.yaml"),
        dst=os.path.join(env.job_config_path_local, "MOO_setting.yaml")
    )
    execute(put_configs, config)
    # now, we delete MOO_setting.yaml file from local config folder in
    # FabFLee/config_files directory
    os.remove(os.path.join(env.job_config_path_local, "MOO_setting.yaml"))

    script = "moo_flee"
    job(dict(script=script))


@task
@load_plugin_env_vars("FabFlee")
# Syntax: fabsim localhost validate_results:flee_conflict_name_localhost_16
def validate_results(output_dir):
    score = vvp_validate_results("{}/{}".format(env.local_results, output_dir))
    #print("Validation {}: {}".format(output_dir.split[-1]), score)
    return score


def make_vvp_mean(np_array, **kwargs):
    #mean_score = np.mean(np_array)
    # convert array to dict
    vvp_dict = {}
    for i in np_array:
        if i[0] in vvp_dict:
            vvp_dict[i[0]].append(i[1])
        else:
            vvp_dict[i[0]] = [i[1]]
    mean_score = 0.0

    means = []
    print("\033[94m--------------------------------")
    print("\033[94m Validation Summary")
    print("--------------------------------\033[0m")


    for l in vvp_dict.keys():
        mean = np.mean(vvp_dict[l])
        dmax = np.max(vvp_dict[l])
        dmin = np.min(vvp_dict[l])
        dstdev = np.std(vvp_dict[l])
        d5 = np.percentile(vvp_dict[l], 5)
        d25 = np.percentile(vvp_dict[l], 25)
        d50 = np.percentile(vvp_dict[l], 50)
        d75 = np.percentile(vvp_dict[l], 75)
        d95 = np.percentile(vvp_dict[l], 95)
        print(f"{l}:")
        print(f"* mean: {mean}, min: {dmin}, max: {dmax}, stdev: {dstdev}, mean error: {mean/2.0}")
        print(f"* 5%: {d5}, 25%: {d25}, 50%: {d50}, 75%: {d75}, 95%: {d95}")
        means.append(mean)

    mean_score = np.mean(means)
    print(f"\033[93mAggregated statistics:")
    print(f"* Mean score: {mean_score}")
    print(f"* Mean error: {mean_score/2.0}\033[0m")
    return mean_score


@task
@load_plugin_env_vars("FabFlee")
# Syntax: fabsim localhost validate_flee_output:<results_dir>
def validate_flee_output(results_dir, mode="rescaled"):
    """
    Goes through all the output directories and calculates the validation
    scores.
    """
    env.flee_validation_mode = mode
    full_results_dir = "{}/{}/RUNS".format(env.local_results, results_dir)
    results = vvp.ensemble_vvp(full_results_dir,
                     vvp_validate_results,
                     make_vvp_mean)

    vresults = results[full_results_dir]

    return vresults


@task
@load_plugin_env_vars("FabFlee")
# Syntax: fabsim localhost validate_flee
def validate_flee(config='validation', simulation_period=0, cores=4, skip_runs=False, label="", mode="rescaled", **args):
    """
    Runs all the validation test and returns all scores + aggregate statistics
    """
    
    env.flee_validation_mode = mode

    if len(label) > 0:
        print("adding label: ", label)
        env.job_name_template += "_{}".format(label)

    clean_fabsim_dirs(config)

    env.prevent_results_overwrite = "delete"

    if not skip_runs:
        pflee_ensemble(config, simulation_period,
                       cores=cores, **args)

    # if not run locally, wait for runs to complete
    update_environment()
    if env.host != "localhost":
        wait_complete("")

    fetch_results()

    results_dir = template(env.job_name_template)
    validate_flee_output(results_dir, mode)


# Commands to create a new conflict
@task
@load_plugin_env_vars("FabFlee")
# Syntax: fabsim localhost new_conflict:<config_name>
def new_conflict(config, **args):
    """
    Create a new conflict configuration directory for Flee simulations.
    """

    local(template("mkdir -p %s/config_files/%s"
                   % (get_plugin_path("FabFlee"), config)))

    local(template("mkdir -p %s/config_files/%s/input_csv"
                   % (get_plugin_path("FabFlee"), config)))

    local(template("mkdir -p %s/config_files/%s/source_data"
                   % (get_plugin_path("FabFlee"), config)))

    local(template("cp %s/flee/config_template/simsetting.yml \
        %s/config_files/%s")
          % (env.flee_location, get_plugin_path("FabFlee"), config))

    local(template("cp %s/flee/config_template/input_csv/sim_period.csv "
                   "%s/config_files/%s/input_csv")
          % (env.flee_location, get_plugin_path("FabFlee"), config))

    local(template("cp %s/flee/config_template/input_csv/closures.csv "
                   "%s/config_files/%s/input_csv")
          % (env.flee_location, get_plugin_path("FabFlee"), config))

    local(template("cp %s/flee/config_template/input_csv/"
                   "registration_corrections.csv "
                   "%s/config_files/%s/input_csv")
          % (env.flee_location, get_plugin_path("FabFlee"), config))

@task
# Syntax: fabsim localhost process_acled:country,dd-mm-yyyy,earliest/fatalities,location/admin2/admin1
def process_acled(country, start_date, filter_option, admin_level):
    """
    Process .csv files sourced from acleddata.com to a <locations.csv> format
    Syntax:
        fabsim localhost process_acled:
        country (e.g. ssudan, mali),
        start_date: "dd-mm-yyyy (date to calculate conflict_date from),
        filter_opt: earliest or fatalities
        **earliest keeps the first occurence of each admin_level,
        fatalities keeps admin_level with the highest fatalities.
        admin_level: is how high the admin level you want to apply the
        filter_opt to i.e. admin1, admin2 or location.
    """
    from .scripts.acled2locations import acled2locations

    acled2locations(
        fab_flee_loc=get_plugin_path("FabFlee"),
        country=country,
        start_date=start_date,
        filter_option=filter_option,
        admin_level=admin_level
    )


@task
@load_plugin_env_vars("FabFlee")
# Syntax: fabsim localhost add_population:<config_name>
def add_population(config, PL="100", CL="100", **args):
    # update_environment(args, {"simulation_period": simulation_period})
    """
    Add population and city limits to the Flee simulation configuration.
    """
    with_config(config)
    if env.machine_name != 'localhost':
        print("Error : This task should only executed on your localhost")
        print("Please re-run is again with :")
        print("\t fabsim localhost add_population:%s" % (config))
        exit()
    env.cityGraph_POPULATION_LIMIT = PL
    env.cityGraph_CITIES_LIMIT = CL
    local("python %s --cityGraph_location %s --API_KEY %s "
          "--POPULATION_LIMIT %s --CITIES_LIMIT %s "
          "--config_location %s --config_name %s"
          % (os.path.join(env.localplugins["FabFlee"],
                          "scripts",
                          "population2locations.py"),
             env.cityGraph_location,
             env.cityGraph_API_KEY,
             env.cityGraph_POPULATION_LIMIT,
             env.cityGraph_CITIES_LIMIT,
             env.job_config_path_local,
             config
             )
          )

@task
# Syntax: fabsim localhost extract_conflict_file:<country_name>,simulation_period=<number>
def extract_conflict_file(config, simulation_period, **args):
    """
    Travels to the input_csv directory of a specific config and extracts
    a conflict progression CSV file from locations.csv.
    """
    # config_dir = "%s/config_files/%s" % (get_plugin_path("FabFlee"), config)
    # local("python3 %s/scripts/location2conflict.py %s \
    #         %s/input_csv/locations.csv %s/input_csv/conflicts.csv"
    #       % (get_plugin_path("FabFlee"),
    #          simulation_period,
    #          config_dir,
    #          config_dir))

    config_dir = os.path.join(
        get_plugin_path("FabFlee"), "config_files", config
    )
    from .scripts.location2conflict import location2conflict
    location2conflict(
        simulation_period=int(simulation_period),
        input_file=os.path.join(config_dir, "input_csv", "locations.csv"),
        output_file=os.path.join(config_dir, "input_csv", "conflicts.csv"),
    )

# Test Functions
# from plugins.FabFlee.test_FabFlee import *

try:
    from plugins.FabFlee.run_simulation_sets import *
except:
    exc_type, exc_value, exc_traceback = sys.exc_info()
    print("Error: failed to import settings module run_simulation_sets")
    pprint(exc_type)
    pprint(exc_value)
    import traceback
    traceback.print_tb(exc_traceback)
    print("The FabFlee run_simulation_sets functionalities are not imported as a result.")
    pass

try:
    # loads Sensitivity analysis (SA) tasks
    from plugins.FabFlee.SA.flee_SA import flee_init_SA
    from plugins.FabFlee.SA.flee_SA import flee_analyse_SA
except:
    exc_type, exc_value, exc_traceback = sys.exc_info()
    print("Error: failed to import settings module flee_SA")
    pprint(exc_type)
    pprint(exc_value)
    import traceback
    traceback.print_tb(exc_traceback)
    print("The FabFlee flee_SA functionalities are not imported as a result.")
    pass


try:
    # loads Automated Visualisation tasks
    from plugins.FabFlee.vis.flee_geo import *
except:
    exc_type, exc_value, exc_traceback = sys.exc_info()
    print("Error: failed to import settings module flee_vis")
    pprint(exc_type)
    pprint(exc_value)
    import traceback
    traceback.print_tb(exc_traceback)
    print("The FabFlee flee_vis functionalities are not imported as a result.")
    pass


try:
    # # loads Validation and Verification Patterns (VVP) tasks
    from plugins.FabFlee.VVP.flee_vvp import flee_init_vvp_LoR
    from plugins.FabFlee.VVP.flee_vvp import flee_analyse_vvp_LoR

    from plugins.FabFlee.VVP.flee_vvp import flee_init_vvp_QoI
    from plugins.FabFlee.VVP.flee_vvp import flee_analyse_vvp_QoI
except:
    exc_type, exc_value, exc_traceback = sys.exc_info()
    print("Error: failed to import settings module flee_vvp")
    pprint(exc_type)
    pprint(exc_value)
    import traceback
    traceback.print_tb(exc_traceback)
    print("The FabFlee flee_vvp functionalities are not imported as a result.")
    pass

try:
    from plugins.FabFlee.run_perf_benchmarks import *
except:
    exc_type, exc_value, exc_traceback = sys.exc_info()
    print("Error: failed to import settings module run_perf_benchmarks")
    pprint(exc_type)
    pprint(exc_value)
    import traceback
    traceback.print_tb(exc_traceback)
    print("The FabFlee run_perf_benchmarks functionalities are not imported as a result.")
    pass
    
try:
    from plugins.FabFlee.refinement import *
except:
    exc_type, exc_value, exc_traceback = sys.exc_info()
    print("Error: failed to import refinement.py command")
    pprint(exc_type)
    pprint(exc_value)
    import traceback
    traceback.print_tb(exc_traceback)
    print("The refinement functionalities are not imported as a result.")
    pass
    
