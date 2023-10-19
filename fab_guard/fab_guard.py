import pandas as pd
import pandera as pa
from pandera import Column, Check, extensions, DataFrameSchema
import os

import plugins.FabFlee.fab_guard.config as config


import functools

# Decorator function
def log(func):
    err_count = 0  # Counter for function calls

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal err_count
        err_count += 1
        result = func(*args, **kwargs)
        log_message = f"Error #{err_count}: {func.__name__} returned {result}\n"

        with open(config.log_file, "a+") as log_file:
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
            df = pd.read_csv(os.path.join(self.input_dir,file), **kwargss)
            self.loaded_files[file]=df
        return df

    def verify(self):
        for key in fgcheck.all:
            fgcheck.all[key](self)

    def register_for_test(self, scheme, input_file):
        df = self.load_file(input_file)
        try:
            scheme.validate(df, lazy=config.lazy)
        except pa.errors.SchemaErrors as err:
            print(err.failure_cases)  # dataframe of schema errors
            # print(err.data)  # invalid dataframe

    def transpose(self, df):
        df = df.T
        # Set the first row as column headers and reset the index
        df.columns = df.iloc[0]
        df = df[1:].reset_index(drop=True)
        # reset the datatype from objevct to int
        return df