# FLEE coupled UQ tutorial

## Prerequisites

This tutorial requires the following:
* Flee
* Flare
* FabSim3
* The FabFlee plugin

### Installing Flee

For installation instructions, see http://www.github.com/djgroen/flee-release

### Installing Flare

For installation instructions, see http://www.github.com/djgroen/flare-release

### Installing FabSim3 and FabFlee

For installation instructions, see http://www.github.com/djgroen/FabSim3/INSTALL

Once you have installed FabSim3, you can install FabFlee by typing `fab localhost install_plugin:FabFlee`.

## Configuration



## Executing the workflow.

To run this workflow in one command, executing first Flare 20 times and then Flee 20 times, just use:
`fab localhost flee_conflict_forecast:mali,simulation_period=50,N=5`

The plot the aggregated output of the simulation ensemble, including confidence intervals, you can use:
`fab localhost plot_uq_output:mali_localhost_16/RUNS,out`
