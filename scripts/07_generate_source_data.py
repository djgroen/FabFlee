"""
Input:
- 'locations.csv': CSV file containing location data including 'name' and 'location_type' columns.

Output:
- 'source_data/data_layout.csv': Maps each camp name to its stub CSV file.
- 'source_data/refugees.csv': Single-entry total refugee count (zero).
- 'source_data/<camp_name>.csv': One stub file per camp (zero count).

Description:
Generates minimal stub validation data files in the 'source_data/' directory for a
given scenario. These files are required by flee's DataTable even when no real
validation data exists (e.g. future projection scenarios). Camp names are read
directly from 'locations.csv' so the stubs stay in sync automatically.

Run this script whenever locations.csv changes.

Usage:
    python 07_generate_source_data.py <scenario_dir>

where <scenario_dir> is the path to the directory containing 'input_csv/' and
'source_data/' subdirectories (e.g. an absolute path or a relative SWEEP directory).

Example:
    python 07_generate_source_data.py /path/to/Iran2026/SWEEP/1A
"""

import csv
import os
import sys


def generate_source_data(scenario_dir):

    scenario_dir = os.path.abspath(scenario_dir)
    # Accept either the scenario root or the input_csv subdirectory directly
    if os.path.basename(scenario_dir) == "input_csv":
        scenario_dir = os.path.dirname(scenario_dir)
    loc_csv = os.path.join(scenario_dir, "input_csv", "locations.csv")
    src_dir = os.path.join(scenario_dir, "source_data")

    if not os.path.exists(loc_csv):
        raise FileNotFoundError(f"locations.csv not found at: {loc_csv}")

    os.makedirs(src_dir, exist_ok=True)

    # Read start date from sim_period.csv if available, else default
    sim_period_csv = os.path.join(scenario_dir, "input_csv", "sim_period.csv")
    start_date = "2000-01-01"
    if os.path.exists(sim_period_csv):
        with open(sim_period_csv, newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                if row and row[0].strip() == "StartDate":
                    start_date = row[1].strip()
                    break

    # Read camp names from locations.csv
    camps = []
    with open(loc_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [k.lstrip('#') for k in reader.fieldnames]
        for row in reader:
            if row.get('location_type', '').startswith('camp'):
                camps.append(row['name'])

    # Write data_layout.csv
    with open(os.path.join(src_dir, "data_layout.csv"), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['total', 'refugees.csv'])
        for camp in camps:
            w.writerow([camp, f'{camp}.csv'])

    # Write refugees.csv
    with open(os.path.join(src_dir, "refugees.csv"), 'w', newline='') as f:
        f.write(f"{start_date},0\n")

    # Write per-camp stub CSV files
    for camp in camps:
        with open(os.path.join(src_dir, f"{camp}.csv"), 'w', newline='') as f:
            f.write(f"{start_date},0\n")

    print(f"Generated source_data for {len(camps)} camps in: {src_dir}")
    print(f"Start date used: {start_date}")
    print("Re-run this script if locations.csv changes.")


scenario_dir = sys.argv[1]
generate_source_data(scenario_dir)
