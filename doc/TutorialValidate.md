Ensemble Output Validation of Multiscale Migration Prediction
======

This tutorial explains a validation pattern, namely ensemble output validation, given an ensemble of output directories. 
It has a sample testing function on each directory, and prints the outputs to screen, and uses an aggregation function to combine all outputs into a compound metric.

To showcase on migration simulations, we use `validate_flee_output` function in FabFlee for a single conflict instance or various conflict instances with replica or ensemble runs.

## Ensemble Validation of a single conflict with multiple instances

1.  To check the existing output directory, simply run
    ```
    cd (FabSim3 Home)/results
    ls
    ```
    In `(FabSim3 Home)/results`, an output directory for a conflict scenario should be present from previous executions, which is most likely called `<conflict name>_localhost_16` or `<conflict name>_<remote machine name>_<number>` (e.g. mali_localhost_16 or mali_eagle_vecma_4). This conflict output directory should have multiple run directories in `RUNS`, as `Run_1`, `Run_2` and so on.
    
2.  To run validation on these output directories of a single conflict with multiple instances, simply go to the FabFlee directory and run
    ```
    fab localhost validate_flee_output:<conflict name>_localhost_16 
    ```    
    or
    ```
    fab <remote machine name> validate_flee_output:<conflict name>_<remote machine name>_<number> 
    ```
    
    For instance, to validate the Mali conflict instance, simply type
    ```
    fab localhost validate_flee_output:mali_localhost_16
    ```
    or
    ```
    fab eagle_vecma validate_flee_output:mali_eagle_vecme_4 
    ```

## Ensemble Validation of a single conflict instance with replicas
1.  To account for stochastic effects in the model, run simulations by specifying multiple replicas:
    ```
    fabsim localhost flee:<conflict_name>,simulation_period=<number>,replicas=<number>
    ```
    or 
    ```
    fabsim <remote machine name> flee:<conflict_name>,simulation_period=<number>,replicas=<number>
    ```
    It will execute a conflict instance with multiple replicas in `(FabSim3 Home)/results`, which are most likely called `<conflict name>_localhost_16_replica_<number>` or `<conflict name>_<remote machine name>_<number>_replica_<number>` (e.g. `mali_localhost_16_replica_1` or `mali_eagle_vecma_4_replica_2`). To run ensemble validation for these replica instances, make directory with subdirectory `<conflict_name>/RUNS` in `(FabSim3 Home)/results` and place replica runs inside. This will provide average over all those replicas for each conflict instance. 
    
    For instance, there are `mali_localhost_16_replica_1` and `mali_localhost_16_replica_2` output directories, which are placed inside `mali/RUNS` for ensemble validation.      
    
2. To run validation on these output directories of a single instance with replica instances, simply go to the FabFlee directory and run
    ```
    fab localhost validate_flee_output:<conflict name>
    ```    
    or
    ```
    fab <remote machine name> validate_flee_output:<conflict name>
    ```
    
    For instance, to validate the Mali conflict instance, simply type
    ```
    fab localhost validate_flee_output:mali
    ```
    or
    ```
    fab eagle_vecma validate_flee_output:mali
    
## Ensemble Validation of multiple conflict instances

1.  To check the existing output directory, simply run
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

