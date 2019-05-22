# FLEE coupled UQ tutorial

## Prerequisites

To perform this tutorial you will require the following installed:
* Flee
* Flare
* FabSim3
* The FabFlee plugin

Below you can find installation instructions for each of these tools.

### Installing Flee

For installation instructions, see http://www.github.com/djgroen/flee-release

### Installing Flare

For installation instructions, see http://www.github.com/djgroen/flare-release

### Installing FabSim3 and FabFlee

For installation instructions, see http://www.github.com/djgroen/FabSim3/INSTALL

Once you have installed FabSim3, you can install FabFlee by typing:

`fab localhost install_plugin:FabFlee`.

## Configuration

## Forced migration simulations: description

### model structure

### model refinement


# Executing simulations of population displacement

### run a single population displacement validation test

`fab localhost flee:mali,simulation_period=300`

### run an ensemble simulation, analyzing variability.

`fab localhost flee_ensemble:mali,simulation_period=300,N=10`

### run a coupled simulation with basic UQ

`fab localhost flee_conflict_forecast:mali,N=2,simulation_period=300`

# Going the next mile (optional content)

### Running the coupled simulation on a supercomputer

`fab eagle flee_conflict_forecast:mali,N=20,simulation_period=300`

### Running an ensemble simulation on a supercomputer using Pilot Jobs

`fab eagle flee_ensemble:mali,N=20,simulation_period=300,Pilot=true`

### Running an ensemble simulation on a supercomputer using Pilot Jobs and QCG Broker

`fab qcg flee_ensemble:mali,N=20,simulation_period=300,Pilot=true`
