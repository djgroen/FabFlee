Ensemble Output Validation of Multiscale Migration Prediction
======

This tutorial explains a validation pattern, namely ensemble output validation, given an ensemble of output directories. 
It has a sample testing function on each directory, and prints the outputs to screen, and uses an aggregation function to combine all outputs into a compound metric.

To showcase on migration simulations, we use `validate_flee_output` function in FabFlee for a single conflict instance or various conflict instances with ensemble runs.

## Ensemble Validation of a single conflict instance

1.  To check existing output directory, simply run
    ```
    cd (FabSim3 Home)/results
    ls
    ```
    In `(FabSim3 Home)/results`, an output directory for a conflict scenario should be present from previous executions, which is most likely called `<conflict name>_localhost_16` or `<conflict name>_<remote machine name>_<number>` (e.g. mali_localhost_16 or mali_eagle_vecma_4). This conflict output directory should have multiple run directories in `RUNS`, as `Run_1`, `Run_2` and so on.
    
2.  To run validation on these output directories of a single instance, simply go to the FabFlee directory and run
    ```
    fab localhost validate_flee_output:<conflict name>_localhost_16 
    ```
    or
    ```
    fab <remote machine name> validate_flee_output:<conflict name>_<remote machine name>_<number> 
    ```
    
    For instance, to validate the Mali conflict instance,simply type
    ```
    fab localhost validate_flee_output:mali_localhost_16
    ```
    or
    ```
    fab eagle_vecma validate_flee_output:mali_eagle_vecme_4 
    ```
    
    
## Ensemble Validation of multiple conflict instances

1.  To check existing output directory, simply run
    ```
    cd (FabSim3 Home)/results
    ls
    ```
    In `(FabSim3 Home)/results`, an output directory for multiple conflict scenarios should be present from previous execution in a directory, which is most likely called `validation_localhost_16` or `validation_<remote machine name>_<number>` (e.g. validation_eagle_vecma_4). This validation output directory should have multiple simulation instances in `RUNS`, as `burundi`, `mali` and other conflict instance names.
    
2.  To run validation on these output directories of multiple conflict instances, simply go to the FabFlee directory and run
    ```
    fab localhost validate_flee_output:validation_localhost_16 
    ```
    or
    ```
    fab <remote machine name> validate_flee_output:validation_<remote machine name>_<number> 
    ```
    
    For instance, to validate multiple conflict instances in the validation directory. simply execute
    ```
    fab eagle_vecma validate_flee_output:validation_eagle_vecme_4 
    ```
