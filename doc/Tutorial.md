# FLEE coupled UQ tutorial

In this tutorial we will explain how you can combine a simple stochastic conflict evolution model (Flare) with an agent-based migration model (Flee), perform a set of runs based on different conflict evolutions, and visualize the migrant arrivals with confidence intervals.

## 1. Prerequisites

To perform this tutorial you will require a Linux environment, and the following software packages:
* Flee
* Flare
* FabSim3
* The FabFlee plugin

Below you can find installation instructions for each of these tools.

### Installing Flee

For this tutorial, you will need to install Flee on your local workstation.
<br/> For installation instructions, see http://www.github.com/djgroen/flee-release

We will assume that you will install Flee in a directory called (Flee Home).

### Installing Flare

For this tutorial, you will also need to install Flare on your local workstation.
<br/> For installation instructions, see http://www.github.com/djgroen/flare-release

We will assume that you will install Flare in a directory called (Flare Home).

### Installing FabSim3 and FabFlee

For installation instructions, see https://github.com/djgroen/FabSim3/blob/master/INSTALL.md

We will assume that you will install FabSim3 in a directory called (FabSim Home).

_NOTE: Please make sure, both `machines.yml` and `machines_user.yml` are configured correctly based on the installation guide._


Once you have installed FabSim3, you can install FabFlee by typing:
```
fab localhost install_plugin:FabFlee
```


## 2. Configuration

Once you have installed the required dependencies, you will need to take a few small configuration steps:
1. Go to `(FabSim Home)/deploy`
2. Open `machines_user.yml`
3. Under the section `default:`, please add the following lines:
   <br/> a. flee_location=(Flee Home) 
   <br/> _NOTE: Please replace (Flee Home) with your actual install directory._
   
   <br/> b. flare_location=(Flare Home)
   <br/> _NOTE: Please replace (Flare Home) with your actual install directory._



# Forced migration simulations

In this section we will show you how you can run different types of migration simulations. We first explain how you can do basic analysis using a single model, and then explain how you can perform a coupled application run that features very basic uncertainty quantification.

## Executing single-model migration simulations

FabFlee comes with a range of sample simulation domains. For instance, a basic model for the 2012 Mali conflict can be found in`(FabSim Home)/plugins/FabFlee/config_files/mali`.

1. To run a single population displacement validation test, using this model, simply type:
```
fab localhost flee:mali,simulation_period=300
```

_NOTE: Please ensure that you reside within the FabSim3 installation directory (or a subdirectory of it), whenever you run any `fab` commands._

2. You can copy back any results from completed runs using:
```
fab localhost fetch_results
```
The results will then be in a directory inside `(FabSim Home)/results`, which is most likely called `mali_localhost_16`.

This is a little redundant for runs on localhost, but essential if you run on any remote machines, so it is good to get into this habit.

3. You can plot the simulation output using:
```
fab localhost plot_output:mali_localhost_16,out
```

## Ensembles

Now you may want to run multiple simulations, to see to what extent the definition of the maximum run speed in Flee affect the overall results. To do so, you can create an ensemble definition.

### Step 1: Duplicate configuration directory

Your main configuration directory for this ensemble is in `config_files/mali`. To create a run speed test, it is best to duplicate this directory by:
```
cp -r (FabFlee Location)/config_files/mali (FabFlee Location)/config_files/mali_runspeed_test
```

### Step 2: Create SWEEP directory

Next, you should create a directory named `SWEEP` inside this directory, e.g. through
```
mkdir (FabFlee Location)/config_files/mali_runspeed_test/SWEEP
```

Inside this SWEEP directory, you can then provide modified input files for each particular run instance by creating a subdirectory for it.

<br/> For instance, to create a run instance with a maximum run speed of 200, we can create a subdirectory called `200`, and create a simsetting.csv file in it with the following contents:`"MaxMoveSpeed",200`

To illustrate **simsetting.csv** file:

|MaxMoveSpeed |200 |
|-------------|----|

You can then create similar directories with inputs that have a run speed of 100, or 400. Or if you're too lazy to do that, just copy the contents of `(FabFlee Location)/config_files/mali/example_sweepdir` to `(FabFlee Location)/config_files/mali_runspeed_test/SWEEP`. 

### Step 3: Run an ensemble of run speed tests

To run the ensemble, you can type:
```
fab localhost flee_ensemble:mali_runspeed_test,simulation_period=300
```

### Step 4: Analyze the output

You can copy back any results from completed runs using:
```
fab localhost fetch_results
```
The results will then be in a directory inside `(FabSim Home)/results` which is most likely called `mali_runspeed_test_localhost_16`.

And you can plot the simulation output using:
```
fab localhost plot_uq_output:mali_runspeed_test_localhost_16,out
```
As a reminder: we use `plot_output` to visualize outputs of a single run, and `plot_uq_output` to collate and visualize results from an ensemble.


## Executing coupled migration simulations

Now a single simulation run is nice, but as this tutorial is part of a project on *multiscale* uncertainty quantification, the least we can offer is a simple UQ analysis using a coupled scenario.

The workflow you will test out here involves the following:
1. We run a set of simple conflict evolution simulations (Flare) in the context of Mali.
2. We gather the conflict evolutions generated by this simulation, and convert them to create input for an ensemble of agent-based migration (Flee) simulation.
3. We run a Flee simulation for each conflict evolution generated.
4. We analyze and visualize a basic result.

### Step 1: Run a Flare ensemble

To run an ensemble of Flare simulations, generating different conflict evolutions, one can simply type:
```
fab localhost flare_ensemble:mali,N=10,simulation_period=300,out_dir=flare-out-scratch
```
This generates a range of CSV files, which you can find in `(FabFlee Home)/results-flare/flare-out-scratch`.


### Step 2: Convert Flare output to Flee input

To convert this output to Flee input you can then use.
```
fab localhost couple_flare_to_flee:mali,flare_out=flare-out-scratch
```
This generates a SWEEP directory in `(FabFlee Home)/config_files/mali`, which in turn contains all the different conflict evolutions.


### Step 3: Run an ensemble of Flee simulations

To then run a Flee ensemble over all the different configurations, simply type:
```
fab localhost flee_ensemble:mali,simulation_period=300
```
Note that for Flee ensembles there is no need to specify the parameter `N`. It simply launches one run for every subdirectory in the `SWEEP` folder.

### Step 4: Analyze the output

You can copy back any results from runs using:
```
fab localhost fetch_results
```
The results will then be in a directory inside `(FabSim Home)/results` which is most likely called `mali_localhost_16`.

Assuming this name, you can then run the following command to generate plots:
```
fab localhost plot_uq_output:mali_localhost_16,out
```
And you can inspect the plots by examining the `out` subdirectory of your results directory.

### Step 1-3 in a one-liner

To run a coupled simulation with basic UQ, and basically repeat steps 1-3 in one go, just type:
```
fab localhost flee_conflict_forecast:mali,N=2,simulation_period=300
```




# Going the next mile (optional content)

### Running the coupled simulation on a supercomputer
```
fab eagle flee_conflict_forecast:mali,N=20,simulation_period=300
```
1. Run `fab eagle job_stat_update` to check if you jobs are finished or not
2. Run `fab eagle fetch_results` to copy back results from `eagle` machine. The results will then be in a directory inside `(FabSim Home)/results`, which is most likely called `mali_eagle_16`
3. Run `fab localhost plot_uq_output:mali_eagle_16,out` to generate plots


<!---
### Running an ensemble simulation on a supercomputer using Pilot Jobs
```
fab qcg flee_ensemble:mali,N=20,simulation_period=300,PilotJob=true
```
-->

### Running an ensemble simulation on a supercomputer using Pilot Jobs and QCG Broker

```
fab qcg flee_ensemble:mali,N=20,simulation_period=300,PilotJob=true
```
1. Run `fab qcg job_stat_update` to check if you jobs are finished or not
2. Run `fab qcg fetch_results` to copy back results from `qcg` machine. The results will then be in a directory inside `(FabSim Home)/results`, which is most likely called `mali_qcg_16`
3. Run `fab localhost plot_uq_output:mali_qcg_16,out` to generate plots

# Acknowledgements

This work was supported by the VECMA and HiDALGO projects, which has received funding from the European Union Horizon 2020 research and innovation programme under grant agreement No 800925 and 824115.
