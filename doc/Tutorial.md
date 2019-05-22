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

Once you have installed FabSim3, you can install FabFlee by typing:

`fab localhost install_plugin:FabFlee`.

## Configuration

## Refugee movement : description

### model structure

### model refinement


# Executing simulations of population displacement

### run a single population displacement validation test

`fab localhost flee:mali,simulation_period=300`

### run an ensemble simulation, analyzing variability.

`fab localhost flee_ensemble:mali,simulation_period=300,N=10`

### run a coupled simulation with basic UQ

`fab localhost flee_conflict_forecast:mali,N=2,simulation_period=300`
