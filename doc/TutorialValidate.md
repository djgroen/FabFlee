Ensemble Output Validation of Multiscale Migration Prediction
======

This tutorial explains a validation pattern, namely ensemble output validation, given an ensemble of output directories. 
It has a sample testing function on each directory, and prints the outputs to screen, and uses an aggregation function to combine all outputs into a compound metric.

To showcase on migration simulations, we use `validate_flee_output` function in FabFlee for a single conflict instance or various conflict instances with replica or ensemble runs.

Throughout the tutorial we use the `fab` command, but it is also possible to use the `fabsim` command if you have set your paths correctly. The latter command will give clearer error reporting when you mistype your instruction.

## Ensemble Validation of a single conflict with multiple instances

1.  To execute multiple instances of a single conflict scenario, simply run the `flee_ensemble` function as follows  
    ```
    fab localhost flee_ensemble:<conflict_name>,simulation_period=<number>
    ```
    or 
    ```
    fab <remote machine name> flee_ensemble:<conflict_name>,simulation_period=<number>
    ```
    > NOTE : A conflict scenario in the `config_files` should contain `SWEEP` directory with other instances for execution. 

    For instance, to execute ensemble simulations for the Ethiopia conflict, simply type
    ```
    fab localhost flee_ensemble:ethiopia,simulation_period=147
    ```
    or
    ```
    fab eagle_vecma flee_ensemble:ethiopia,simulation_period=147
    ```
    
2.  Copy back any results from completed runs using:
    ```
    fab localhost fetch_results
    ```
    To check the executed output directory, simply run
    ```
    cd (FabSim3 Home)/results
    ls
    ```
    In `(FabSim3 Home)/results`, an output directory for a conflict scenario should be present from previous execution, which is most likely called `<conflict name>_localhost_16` or `<conflict name>_<remote machine name>_<number>` (e.g. ethiopia_localhost_16 or ethiopia_eagle_vecma_4). This conflict output directory should have multiple run directories in `RUNS`. 

3.  To run validation on these output directories of a single conflict with multiple instances, simply go to the FabFlee directory and run
    ```
    fab localhost validate_flee_output:<conflict name>_localhost_16 
    ```    
    or
    ```
    fab <remote machine name> validate_flee_output:<conflict name>_<remote machine name>_<number of cores used> 
    ```
    
    For instance, to validate the Ethiopia conflict instance, simply type
    ```
    fab localhost validate_flee_output:ethiopia_localhost_16
    ```
    or
    ```
    fab eagle_vecma validate_flee_output:ethiopia_eagle_vecma_4 
    ```

## Ensemble Validation of a single conflict instance with replicas

1.  To account for stochastic effects in the model, run simulations by specifying multiple replicas:
    ```
    fab localhost flee:<conflict_name>,simulation_period=<number>,replicas=<number>
    ```
    or 
    ```
    fab <remote machine name> flee:<conflict_name>,simulation_period=<number>,replicas=<number>
    ```
    For instance, to execute replica simulations for the Mali conflict, simply type
    ```
    fab localhost flee:mali,simulation_period=300,replicas=2
    ```
    or
    ```
    fab eagle_vecma flee:mali,simulation_period=300,replicas=2
    ```
    
2.  Copy back any results from completed runs using:
    ```
    fab localhost fetch_results
    ```
    It will copy output directory of a conflict instance with multiple replicas to `(FabSim3 Home)/results`, which are most likely called `<conflict name>_localhost_16_replica_<number>` or `<conflict name>_<remote machine name>_<number>_replica_<number>` (e.g. `mali_localhost_16_replica_1` or `mali_eagle_vecma_4_replica_2`). 
    
3.  To run ensemble validation for these replica instances, make directory with subdirectory `<conflict_name>/RUNS` in `(FabSim3 Home)/results` and place replica runs inside. This will provide average over all those replicas for each conflict instance. 
    
    For instance, there are `mali_localhost_16_replica_1` and `mali_localhost_16_replica_2` output directories, which are placed inside `mali/RUNS` for ensemble validation.      
    
4.  To run validation on these output directories of a single instance with replica instances, simply go to the FabFlee directory and run
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

To execute multiple conflict scenarios, simply run the `validate_flee` function as follows  
```
fab localhost validate_flee
```
or 
```
fab <remote machine name> validate_flee
```
which executes multiple simulations in `FabFlee/config_files/validation/SWEEP` containing `burundi2015`, `car2013` and other conflict instances.
    
After simulations are completed, validation values of these conflict instances are calculated and printed to the terminal screen. 
