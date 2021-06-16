import sys
import os
import yaml
import argparse
import numpy as np
import pandas as pd
import csv
import random
import glob
from statistics import mean
from pprint import pprint, pformat


from pymoo.util.misc import stack
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.algorithms.nsga3 import NSGA3
from pymoo.algorithms.moead import MOEAD
from pymoo.factory import get_sampling, get_crossover, get_mutation, \
    get_problem, get_reference_directions
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.model.problem import Problem
from pymoo.factory import get_performance_indicator


work_dir = os.path.dirname(os.path.abspath(__file__))
EXEC_LOG_FILE = None


class dict_to_obj:

    def __init__(self, in_dict: dict):
        assert isinstance(in_dict, dict)
        for key, val in in_dict.items():
            if isinstance(val, (list, tuple)):
                setattr(self, key, [dict_to_obj(x) if isinstance(
                    x, dict) else x for x in val])
            else:
                setattr(self, key, dict_to_obj(val)
                        if isinstance(val, dict) else val)


def MOO_log(msg):
    with open(EXEC_LOG_FILE, "a") as log_file:
        print("{}".format(msg), file=log_file)


def read_MOO_setting_yaml():
    """
    read MOO setting from yaml file
    """
    with open(os.path.join(work_dir, "MOO_setting.yaml")) as f:
        MOO_CONFIG = yaml.safe_load(f)

    # convert the json to a nested object
    # MOO_CONFIG_DICT = dict_to_obj(MOO_CONFIG)
    # return MOO_CONFIG_DICT
    return MOO_CONFIG


class FLEE_MOO_Problem(Problem):

    def __init__(self, execution_mode, simulation_period, cores,
                 work_dir=work_dir):

        # TODO: add input vraibles to MOO_setting.yaml file
        super().__init__(n_var=2,
                         n_obj=3,
                         xl=np.array([-500, -500]),
                         xu=np.array([600, 600]))
        self.work_dir = work_dir
        self.cnt_SWEEP_dir = 0
        self.execution_mode = execution_mode
        self.simulation_period = simulation_period
        self.cores = cores

    def avg_distance(self, agents_out_files, camp_name):

        df_array = [pd.read_csv(filename, index_col=None, header=0)
                    for filename in agents_out_files]

        df = pd.concat(df_array, axis=0, ignore_index=True)

        # filter rwos for agent location == camp_name
        df = df[(df["agent location"] == camp_name) &
                (df["distance_moved_this_timestep"] > 0)
                ]

        df.to_csv(os.path.join(
            os.path.dirname(agents_out_files[0]), "df_agents.out.csv"),
            sep=",",
            mode="w",
            index=False,
            encoding='utf-8'
        )

        return df["distance_travelled"].mean()

    def find_closest_location_to_camp(self, x_coordinate, y_coordinate):
        # (1) create conflict zone A, towns B, C, D
        locations = ['A', 'B', 'C', 'D']
        coordinate = np.array([[0, 100], [100, 100], [100, 0], [0, 0]])

        # (3) calculate the distance between Z and A, B, C, D, respectively.
        dist = []
        for coord in coordinate:
            squared = (coord[0] - x_coordinate)**2 + \
                (coord[1] - y_coordinate)**2
            dist.append(np.sqrt(squared))

        # (4) find the shortest path
        min_dist = np.amin(dist)
        index_min_dist = dist.index(min_dist)
        nearest_loc = locations[index_min_dist]

        return nearest_loc, min_dist

    def change_route_to_camp(self, csv_name):
        """
        Change the location that connect to the camp
        """
        MOO_log(msg="\n[change_route_to_camp]")
        # Read the data in coordinates.csv file row by row.
        reader = csv.reader(
            open(os.path.join(self.work_dir, "input_csv", csv_name))
        )

        rows = [l for l in reader]
        rows.pop(0)  # remove header from rows
        for row in rows:
            x_cord = row[0]
            y_cord = row[1]
            MOO_log(msg="\tx_cord ={}".format(x_cord))
            MOO_log(msg="\ty_cord ={}".format(y_cord))

            # 1. Find the nearest location to camp and calculate the distance
            # between them.
            nearest_loc, min_dist = self.find_closest_location_to_camp(
                x_coordinate=float(x_cord), y_coordinate=float(y_cord)
            )

            # 2. Read routes.csv and modify the data (i.e., the nearest
            # location to camp and the distance between them)
            reader = csv.reader(
                open(os.path.join(self.work_dir, "input_csv", "routes.csv"))
            )

            lines = [l for l in reader]
            for i in range(1, len(lines)):
                if lines[i][1].strip() != 'Z':
                    continue
                lines[i][0] = nearest_loc
                lines[i][2] = str(min_dist)
                # print(lines[i])

            MOO_log(msg="\tx coodinate of camp Z: {} \n\t"
                    "y coodinate of camp Z: {}\n\t"
                    "nearest location: {}\n\t"
                    "distance to {}:{}".format(
                        float(x_cord),
                        float(y_cord),
                        nearest_loc,
                        nearest_loc, min_dist)
                    )

            # 3. Write the updated route.csv in the moo_f1_c1_t3 SWEEP
            # directory.
            sweep_dir = os.path.join(self.work_dir, "SWEEP")
            # curr_dir_count = len(os.listdir(sweep_dir))
            curr_dir_count = self.cnt_SWEEP_dir
            sub_dir_SWEEP = os.path.join(
                sweep_dir, "{}".format(curr_dir_count + 1), "input_csv"
            )

            if os.path.exists(sub_dir_SWEEP):
                raise RuntimeError(
                    "SWEEP dir {} is exists !!!!!".format(sub_dir_SWEEP)
                )

            os.makedirs(sub_dir_SWEEP)
            MOO_log(msg="\tgenerates SWEEP : {}".format(sub_dir_SWEEP))

            writer = csv.writer(
                open(os.path.join(sub_dir_SWEEP, "routes.csv"), "w")
            )
            writer.writerows(lines)

            self.cnt_SWEEP_dir += 1
            MOO_log(msg="\t{}".format("-" * 30))

    def flee_optmization(self, run_dir, camp_name):
        MOO_log(msg="\n[flee_optmization] called for "
                "run_dir = {} camp_name = {}".format(run_dir, camp_name)
                )

        # calculate Camp population
        df = pd.read_csv(os.path.join(run_dir, "out.csv"))

        sim_camp_population_last_day = df["{} sim".format(camp_name)].iloc[-1]
        sim_camp_population = df["{} sim".format(camp_name)].tolist()

        MOO_log(msg="\tsim camp {} population of the last day = {}".format(
            camp_name, sim_camp_population_last_day)
        )
        MOO_log(msg="\tsim camp {} population = {}".format(
            camp_name, sim_camp_population)
        )

        # find the agents.out files
        agents_out_files = glob.glob(
            "{}".format(os.path.join(run_dir, "agents.out.*"))
        )

        # obj#1
        avg_distance_travelled = self.avg_distance(
            agents_out_files=agents_out_files, camp_name=camp_name
        )
        MOO_log(
            msg="\tInput file : {}"
            "\n\t\tavg distance travelled for agents "
            "to camp name {} = {}".format(
                [os.path.basename(filename) for filename in agents_out_files],
                camp_name,
                avg_distance_travelled
            )
        )

        # calculate camp capacity , obj#2
        df = pd.read_csv(os.path.join(run_dir, "input_csv", "locations.csv"))
        camp_population = df[df["#name"] == camp_name]["population"].values[0]
        MOO_log(msg="\tmax camp {} population = {}".format(
            camp_name, camp_population)
        )

        # calculate average remain camp capacity over simulation days, obj#3
        remain_camp_capacity = mean(
            [camp_population - i for i in sim_camp_population]
        )
        MOO_log(msg="\tremain camp {} capacity = {}".format(
            camp_name, remain_camp_capacity)
        )

        # return values  [obj#1, obj#2, obj#3]
        return [avg_distance_travelled, sim_camp_population_last_day,
                remain_camp_capacity]

    def _evaluate(self, population, out, *args, **kwargs):
        """
        1. The _evaluate method takes a two-dimensional NumPy array X
        with n rows and m columns as an input.

        The row represents an individual, namely, a possible location of the
        camp. Each column represents an optimization variable, namely,
        x coordinate and y coordinate of camp, respectively.

        After doing the necessary calculations, the objective values must be
        added to the dictionary, out, with the key F.
        """
        pop_size = len(population)
        MOO_log(
            msg="\n{}\nExecuting _evaluate function with input "
                "population : \n{}\n".format("-" * 30, pformat(population))
        )

        n = 1
        for row in population:
            MOO_log("\tpotential location {}: {}".format(n, row))
            # move the location of camp to the boundary
            # if x coordinate is in the range [0,100]
            # while y coordinate is in the range [0,100]
            if (row[0] >= 0) & (row[0] <= 100) &\
                    (row[1] >= 0) & (row[1] <= 100):
                if random.random() > 0.5:
                    row[0] = random.choice([-500, 600])
                    print("\t-> move to location: {}\n".format(row))
                else:
                    row[1] = random.choice([-500, 600])
                    print("\t-> move to location: {}\n".format(row))
            n += 1

        coordinates_csv_PATH = os.path.join(
            self.work_dir, "input_csv", "coordinates.csv"
        )
        with open(coordinates_csv_PATH, "w", newline="") as file:
            writer = csv.writer(file, delimiter=",")
            writer.writerow(["x_cord", "y_cord"])  # header
            writer.writerows(population)

        # count the number of run folder in SWEEP dir
        sweep_dir = os.path.join(self.work_dir, "SWEEP")

        # Step 2: Run change_route_to_camp function to update the
        # routes.csv file according to the parameter ind, which is
        # the coordinate of camp.
        cnt_SWEEP_dir_before = self.cnt_SWEEP_dir
        self.change_route_to_camp(csv_name="coordinates.csv")

        # Step 3: copy required input to each new generated SWEEP dir and
        # run simulation per each SWEEP dir
        for i in range(cnt_SWEEP_dir_before, self.cnt_SWEEP_dir):
            dest_SWEEP_dir = os.path.join("SWEEP", str(i + 1))
            # list of files and folders to be included
            sel_files_folders = ["**input_csv/***", "**source_data/***",
                                 "run.py",
                                 "run_par.py", "simsetting.csv"
                                 ]
            # Note: be careful with rync command arguments
            rync_cmd = " ".join([
                *["rsync -pthrvz --ignore-existing"],
                *["--include='{}' ".format(sel) for sel in sel_files_folders],
                *["--exclude='*'"],
                *["--exclude='SWEEP'"],
                *["./ {}".format(dest_SWEEP_dir)]
            ])

            MOO_log(msg="\ncopy input files to {}\nrync_cmd = {}".format(
                dest_SWEEP_dir, rync_cmd)
            )
            os.system(rync_cmd)

            # Execute simulation run.py or run_par.py
            if self.execution_mode.lower() == "serial":
                flee_exec_cmd = "python3 run.py input_csv source_data " \
                    "{} simsetting.csv > out.csv".format(
                        self.simulation_period)
            elif self.execution_mode.lower() == "parallel":
                flee_exec_cmd = "mpirun -np {} " \
                    "python3 run_par.py input_csv source_data " \
                    "{} simsetting.csv > out.csv".format(
                        self.cores,
                        self.simulation_period)
            else:
                raise RuntimeError(
                    "The input execution_mode {} not valid!".format(
                        self.execution_mode)
                )

            MOO_log(msg="change dir to -> {}".format(dest_SWEEP_dir))
            os.chdir(dest_SWEEP_dir)
            MOO_log(msg="execute : {}".format(flee_exec_cmd))
            os.system(flee_exec_cmd)

            # clean the SWEEP dir after simulation finished
            clean_cmd = "find . -type f ! \( -name 'out.csv' " \
                "-o -name 'routes.csv' -o -name 'agents.out.*' " \
                "-o -name 'locations.csv' \) -exec rm -rf {} \; ;" \
                "rm -rf source_data"
            os.system(clean_cmd)

            # back to root simulation job dir
            os.chdir(self.work_dir)

        # Step 3: Calculate objective values
        # Create an csv file only contains header
        with open("objectives.csv", "w", newline="") as file:
            writer = csv.writer(file, delimiter=",")
            # add header
            writer.writerow(["Objective #1", "Objective #2", "Objective #3"])

        # Calculate objective values and save the data in objectives.csv file
        for i in range(cnt_SWEEP_dir_before, self.cnt_SWEEP_dir):
            dest_SWEEP_dir = os.path.join("SWEEP", str(i + 1))
            row = self.flee_optmization(run_dir=dest_SWEEP_dir, camp_name="Z")
            with open("objectives.csv", "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(row)

        MOO_log(msg="=" * 50)
        # Fetch the objective values
        objectives = pd.read_csv("objectives.csv")
        MOO_log(msg="objectives.csv =\n{}".format(pformat(objectives)))

        # objective 1: minimize average distance travelled by each arriving
        # refugee.
        f1 = objectives["Objective #1"].values
        MOO_log(msg="\tf1: {}".format(f1))

        # objective 2: maximize camp population, i.e.,the number of people in
        # the camp at the end of the simulation.
        f2 = -objectives["Objective #2"].values
        MOO_log(msg="\tf2: {}".format(f2))

        # objective 3: minimize camp capacity – camp population.
        f3 = objectives["Objective #3"].values
        MOO_log(msg="\tf3: {}".format(f3))

        MOO_log(msg="=" * 50)

        out["F"] = np.column_stack([f1, f2, f3])


if __name__ == "__main__":
    # Instantiate the parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution_mode", action="store", default="serial")
    parser.add_argument("--simulation_period", action="store", type=int,
                        default="-1")
    parser.add_argument("--exec_log_file", action="store",
                        default="log_MOO.txt")

    parser.add_argument("--cores", action="store", type=int, default="1")

    args = parser.parse_args()

    execution_mode = args.execution_mode
    simulation_period = args.simulation_period
    cores = args.cores
    EXEC_LOG_FILE = os.path.join(work_dir, args.exec_log_file)

    MOO_log(msg="run_MOO input args : {}".format(args))

    # read MOO setting from config yaml file
    MOO_CONFIG = read_MOO_setting_yaml()
    MOO_log(msg="MOO_CONFIG =\n{}".format(pformat(MOO_CONFIG)))

    problem = FLEE_MOO_Problem(
        execution_mode=execution_mode,
        simulation_period=simulation_period,
        cores=cores,
    )

    algorithm = None

    alg_name = MOO_CONFIG["alg_name"]

    crossover_func = MOO_CONFIG["crossover_func"]
    crossover_func_args = MOO_CONFIG["crossover_func_args"][crossover_func]

    mutation_func = MOO_CONFIG["mutation_func"]
    mutation_func_args = MOO_CONFIG["mutation_func_args"][mutation_func]

    alg_specific_args = MOO_CONFIG["alg_specific_args"][alg_name]

    try:
        ref_dir_func = alg_specific_args["ref_dir_name"]
        ref_dir_func_args = MOO_CONFIG["ref_dir_func"][ref_dir_func]
        ref_dir_func_args.update({"n_dim": problem.n_obj})
    except KeyError as e:
        # DO NOT raise any Exception if the alg_name does not require
        # any input reference direction function
        pass
    except Exception as e:
        print(e)
        sys.exit()

    if alg_name == "NSGA2":
        sampling_func = MOO_CONFIG["sampling_func"]
        pop_size = alg_specific_args["pop_size"]
        #################
        # set algorithm #
        #################
        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=get_sampling(sampling_func),
            crossover=get_crossover(crossover_func, **crossover_func_args),
            mutation=get_mutation(mutation_func, **mutation_func_args),
            eliminate_duplicates=True
        )
        #####################
        # algorithm logging #
        #####################
        MOO_log(
            msg="algorithm = {}(\n"
            "pop_size={},\n"
            "sampling=get_sampling({}),\n"
            "crossover=get_crossover({},{}),\n"
            "mutation=get_mutation({},{}),\n"
            "eliminate_duplicates=True\n"
            ")".format(
                alg_name,
                pop_size,
                sampling_func,
                crossover_func, crossover_func_args,
                mutation_func, mutation_func_args,
            )
        )

    elif alg_name == "MOEAD":
        alg_specific_args = MOO_CONFIG["alg_specific_args"]["MOEAD"]
        n_neighbors = alg_specific_args["n_neighbors"]
        decomposition = alg_specific_args["decomposition"]
        prob_neighbor_mating = alg_specific_args["prob_neighbor_mating"]
        #################
        # set algorithm #
        #################
        algorithm = MOEAD(
            ref_dirs=get_reference_directions(ref_dir_func,
                                              **ref_dir_func_args),
            n_neighbors=n_neighbors,
            decomposition=decomposition,
            prob_neighbor_mating=prob_neighbor_mating,
            crossover=get_crossover(crossover_func, **crossover_func_args),
            mutation=get_mutation(mutation_func, **mutation_func_args),
        )
        #####################
        # algorithm logging #
        #####################
        MOO_log(
            msg="algorithm = {}(\n"
            "ref_dirs = get_reference_directions({},{}),\n"
            "n_neighbors = {}\n"
            "decomposition = {}\n"
            "prob_neighbor_mating = {}\n"
            "crossover=get_crossover({},{}),\n"
            "mutation=get_mutation({},{}),\n"
            ")".format(
                alg_name,
                ref_dir_func, ref_dir_func_args,
                n_neighbors,
                decomposition,
                prob_neighbor_mating,
                crossover_func, crossover_func_args,
                mutation_func, mutation_func_args,
            )
        )

    elif alg_name == "NSGA3":
        pop_size = alg_specific_args["pop_size"]
        #################
        # set algorithm #
        #################
        algorithm = NSGA3(
            pop_size=pop_size,
            ref_dirs=get_reference_directions(ref_dir_func,
                                              **ref_dir_func_args),
            crossover=get_crossover(crossover_func, **crossover_func_args),
            mutation=get_mutation(mutation_func, **mutation_func_args),
        )
        #####################
        # algorithm logging #
        #####################
        MOO_log(
            msg="algorithm = {}(\n"
            "pop_size = {}\n`"
            "ref_dirs = get_reference_directions({},{}),\n"
            "crossover=get_crossover({},{}),\n"
            "mutation=get_mutation({},{}),\n"
            ")".format(
                alg_name,
                pop_size,
                ref_dir_func, ref_dir_func_args,
                crossover_func, crossover_func_args,
                mutation_func, mutation_func_args,
            )
        )

    if algorithm is None:
        raise RuntimeError(
            "Input alg_name = {} is not valid or "
            "not supported within run_MOO.py".format(
                MOO_CONFIG.alg_name)
        )

    # convert dict {'n_gen': 2}} to tuple ('n_gen', 2)
    termination = list(MOO_CONFIG["termination"].items())[0]
    MOO_log(msg="termination = {}".format(termination))

    res = minimize(
        problem=problem,
        algorithm=algorithm,
        termination=termination,
        verbose=True
    )

    MOO_log(msg="{}".format("#" * 50))
    MOO_log(msg="optimal locations of camp Z:\n\t{}".format(res.X))
    MOO_log(msg="corresponding objective values:\n\t{}".format(res.F))

    out_X = res.X
    out_F = res.F
    out_F[:, 1] = -out_F[:, 1]
    output = np.hstack([out_X, out_F])
    with open("population.csv", "w", newline="") as file:
        writer = csv.writer(file, delimiter=",")
        writer.writerow(["x_cord", "y_cord", "obj_1",
                         "obj_2", "obj_3"])  # header
        writer.writerows(output)
    MOO_log(msg="The output is stored in {}/population.csv".format(work_dir))
