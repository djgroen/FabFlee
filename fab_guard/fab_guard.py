import pandas as pd
import pandera as pa
import datetime
from pandera import Column, Check, extensions, DataFrameSchema
import os

import plugins.FabFlee.fab_guard.config as config
import functools

err_count = 0
# Decorator function
def log(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        global err_count
        err_count += 1
        result = func(*args, **kwargs)
        log_message = f"Error #{err_count}: {func.__name__} returned {result}\n"
        config_dir = FabGuard.get_instance().input_dir
        log_file_name = os.path.join(config_dir, '..', config.log_file)
        if err_count==1:
            print_msg = f"Read the full error messages from truncated errors in the log file:\n #{log_file_name}"
            print(print_msg)
            with open(log_file_name, "w+") as log_file:
                log_file.write("Timestamp: %s \n" % datetime.datetime.now())
        with open(log_file_name, "a+") as log_file:
            log_file.write(log_message)
            log_file.write("\n========================\n")
        return result

    return wrapper


def makeRegistrar():
    registry = {}
    def registrar(func):
        registry[func.__name__] = func
        return func  # normally a decorator returns a wrapped function,
    # but here we return func unmodified, after registering it
    registrar.all = registry
    return registrar

fgcheck = makeRegistrar()

class FabGuard():
    _instance = None

    def __new__(cls, input_dir):
        if cls._instance is None:
            cls._instance = super(FabGuard, cls).__new__(cls)
            cls._instance.input_dir = input_dir
        return cls._instance

    def __init__(self, input_dir):
        self.input_dir = input_dir
        self.loaded_files = {}

    @staticmethod
    def get_instance():
        return FabGuard._instance

    def load_file(self, file,**kwargss):
        if file in self.loaded_files:
            return self.loaded_files[file]
        else:
            df = pd.read_csv(os.path.join(self.input_dir,file),**kwargss)
            #if (df.iloc[1].str.startswith("#")):
            first_column = df.columns[0]
            if df.columns[0].startswith('#'):
                # Remove the first character aand any trailing quotes from start and end
                new_column_name = first_column.lstrip("#").lstrip('\"').rstrip('\"')
                df = df.rename(columns={first_column: new_column_name})
            self.loaded_files[file]=df
        return df

    # Executes all files that are decorated with the fgcheck decorator
    def verify(self):
        for key in fgcheck.all:
            fgcheck.all[key](self)

    def register_for_test(self, scheme, input_file):
        df = self.load_file(input_file)
        try:
            scheme.validate(df, lazy=config.lazy)
        except pa.errors.SchemaErrors as err:
            print(err.failure_cases)  # dataframe of schema errors
            #print(err.data)  # invalid dataframe

    # Transposes a given dataframe
    def transpose(self, df):
        df = df.T
        # Set the first row as column headers and reset the index
        df.columns = df.iloc[0]
        df = df[1:].reset_index(drop=True)
        # reset the datatype from objevct to int
        return df