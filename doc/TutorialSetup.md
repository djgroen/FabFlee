Setup activities for the FLEE tutorial
=====

This document describes what you need to do to set up all the software required for the Flee tutorial.

## Prerequisites

To perform this tutorial, you will require 
* Linux environment
* Python3
* Python libraries
   * numpy (see https://www.numpy.org)
   * pandas (see https://pandas.pydata.org)
   * matplotlib (see https://matplotlib.org)
* The following software packages:
   * Flee
   * Flare
   * FabSim3
   * The FabFlee plugin

Below you can find installation instructions for each of these packages.

### Installing Flee

To install Flee on your local workstation, you need to clone the Flee repository (see http://www.github.com/djgroen/flee-release):
``` 
git clone https://github.com/djgroen/flee-release.git
```
We will assume that you will install Flee in a directory called (Flee Home), e.g. `~/flee-release/`

### Installing Flare

To install Flare on your local workstation, you need to clone the Flare repository (see http://www.github.com/djgroen/flare-release):
```
git clone https://github.com/djgroen/flare-release.git
```
We will assume that you will install Flare in a directory called (Flare Home), e.g. `~/flare-release/`.

### Installing FabSim3 and FabFlee

To install FabSim3, you need to install dependencies and clone the FabSim3 repository.
<br/> For detailed installation instructions, see https://github.com/djgroen/FabSim3/blob/master/INSTALL.md
```
git clone https://github.com/djgroen/FabSim3.git
```
We will assume that you will install FabSim3 in a directory called (FabSim3 Home), e.g. `~/FabSim3/`.

_NOTE: Please make sure both `machines.yml` and `machines_user.yml` are configured correctly based on the installation guide._


Once you have installed FabSim3, you can install FabFlee by typing:
```
fab localhost install_plugin:FabFlee
```
The FabFlee plugin will appear in `~/FabSim3/plugins/FabFlee`.


## 2. Configuration

Once you have installed the required dependencies, you will need to take a few small configuration steps:
1. Go to `(FabSim Home)/deploy`
2. Open `machines_user.yml`
3. Under the section `default:`, please add the following lines:
   <br/> a. `  flee_location=(Flee Home)`
   <br/> _NOTE: Please replace (Flee Home) with your actual install directory._
   <br/> b. `  flare_location=(Flare Home)`
   <br/> _NOTE: Please replace (Flare Home) with your actual install directory._
   
 ## 3. Main tutorial
 
 Once you have completed these tasks, you can do the main tutorial at https://github.com/djgroen/FabFlee/blob/master/doc/Tutorial.md
