# -*- coding: utf-8 -*-
#
# This source file is part of the FabSim software toolkit, which is 151distributed under the BSD 3-Clause license.
# Please refer to LICENSE for detailed information regarding the licensing.
#
# This file contains FabSim definitions specific to FabFlee.

try:
    from fabsim.base.fab import *
except ImportError:
    from base.fab import *

try:
    from fabsim.plugins.FabFlee.FabFlee import *
except ImportError:
    from plugins.FabFlee.FabFlee import *


# Add local script, blackbox and template path.
add_local_paths("FabFlee")

## TEST FUNCTIONS

env.fabflee_root = "{}/plugins/FabFlee".format(env.localroot)

def pr_utest(test_name, test_outcome_bool):
  # print result of unit tests.
  if test_outcome_bool:
    print("%s has succeeded." % test_name)
    pass
  else:
    print("%s has failed." % test_name)


@task
def test_load_conflict():    # fab localhost test_load_conflict
  # write a function that tests whether this worked! It should return FALSE when it doesn't, TRUE when it does.

  os.system("rm -rf %s/conflict_data/active_conflict" % (env.fabflee_root))
  load_conflict("ABC")

  # True if active_conflict exists and False if it does not
  test_result = os.path.exists("%s/conflict_data/active_conflict/" % (env.fabflee_root))

  pr_utest("test_load_conflict-active_conflict_exists",test_result)


  # True if locations.csv in ABC is identical to locations.csv in active_conflict folder.
  import filecmp
  test_result = filecmp.cmp("%s/conflict_data/ABC/locations.csv" % (env.fabflee_root), "%s/conflict_data/active_conflict/locations.csv" % (env.fabflee_root))

  pr_utest("test_load_conflict-match_csv_files",test_result)


@task
def test_change_capacities():    # fab localhost test_change_capacities
  # write a function that tests whether this worked! It should return FALSE when it doesn't, TRUE when it does.

  os.system("rm -rf %s/conflict_data/active_conflict" % (env.fabflee_root))
  load_conflict("ABC")

  change_capacities(Z="10")

  import csv
  r = csv.reader(open("%s/conflict_data/active_conflict/locations.csv" % (env.fabflee_root), "r"))
  lines = [l for l in r]
  test_result = False

  for i in range(1,len(lines)):
    print(lines[i][0],lines[i][5],lines[i])
    if (lines[i][0].strip() == "Z" and lines[i][7].strip() == "10"):
      test_result = True

  pr_utest("test_add_camp-match_capacity_change",test_result)


@task
def test_add_camp():    # fab localhost test_add_camp
  # write a function that tests whether this worked! It should return FALSE when it doesn't, TRUE when it does.

  os.system("rm -rf %s/conflict_data/active_conflict" % (env.fabflee_root))
  load_conflict("ABC")

  add_camp("Y","YY","YYY","0.00001","1.00000")

  # count the number of lines to see if the extra line has been added.
  import csv
  r = open("%s/conflict_data/active_conflict/locations.csv" % (env.fabflee_root), "r")
  row_count = sum(1 for row in r)
  if row_count == 6:
    test_result = True
  else:
    test_result = False

  pr_utest("test_add_camp-count_raw",test_result)

  # check whether there is a line that matches the contents that you would expect to have been added.
  r.seek(0)
  test_result = False
  for row in r:
    if row.strip() == "Y,YY,YYY,0.00001,1.00000,camp":
      test_result = True

  pr_utest("test_add_camp-match_newline",test_result)


@task
def test_delete_location():      # fab localhost test_delete_location
  # write a function that tests whether this worked! It should return FALSE when it doesn't, TRUE when it does.

  os.system("rm -rf %s/conflict_data/active_conflict" % (env.fabflee_root))
  load_conflict("ABC")

  delete_location("C")

  import csv
  r = csv.reader(open("%s/conflict_data/active_conflict/locations.csv" % (env.fabflee_root), "r"))
  lines = [l for l in r]
  test_result = False

  for i in range(1,len(lines)):
    if lines[i][0].strip() != "C":
      test_result = True

  pr_utest("test_delete_location-check_deleted_line",test_result)


@task
def test_close_camp():			    # fab localhost test_close_camp
  # write a function that tests whether this worked! It should return FALSE when it doesn't, TRUE when it does.

  os.system("rm -rf %s/conflict_data/active_conflict" % (env.fabflee_root))
  load_conflict("ABC")

  close_camp("Z","ZZZ")

  import csv
  r = open("%s/conflict_data/active_conflict/closures.csv" % (env.fabflee_root), "r")
  test_result = False

  for row in r:
    if row.strip() == "location,Z,ZZZ,0,-1":
      test_result = True

  pr_utest("test_close_camp-check_camp_closure",test_result)


@task
def test_close_border():                   # fab localhost test_close_border
  # write a function that tests whether this worked! It should return FALSE when it doesn't, TRUE when it does.

  os.system("rm -rf %s/conflict_data/active_conflict" % (env.fabflee_root))
  load_conflict("ABC")

  close_border("ABC","ZZZ")

  import csv
  r = open("%s/conflict_data/active_conflict/closures.csv" % (env.fabflee_root), "r")
  test_result = False

  for row in r:
    if row.strip() == "country,ABC,ZZZ,0,-1":
      test_result = True

  pr_utest("test_close_border-check_border_closure",test_result)


@task
def test_redirect():		    # fab localhost test_redirect
  # write a function that tests whether this worked! It should return FALSE when it doesn't, TRUE when it does.

  os.system("rm -rf %s/conflict_data/active_conflict" % (env.fabflee_root))
  load_conflict("ABC")

  redirect("Y","Z")

  import csv
  r = csv.reader(open("%s/conflict_data/active_conflict/locations.csv" % (env.fabflee_root), "r"))
  lines = [l for l in r]
  test_result = False

  for i in range(1,len(lines)):
    #print(lines[i][0],lines[i][5],lines[i])
    if (lines[i][0].strip() == "Y" and lines[i][5].strip() == "forwarding_hub"):
      test_result = True

  pr_utest("test_redirect-check_location_type",test_result)


  r = csv.reader(open("%s/conflict_data/active_conflict/routes.csv" % (env.fabflee_root), "r"))
  lines = [l for l in r]
  test_result = False

  for i in range(1,len(lines)):
    #print(lines[i][0],lines[i][3],lines[i])
    if (lines[i][0].strip() == "Y" and lines[i][3].strip() == "2"):
      test_result = True

  pr_utest("test_redirect-check_forced_redirection",test_result)


@task
def test_clear_active_conflict():     # fab localhost test_clear_active_conflict
  # write a function that tests whether this worked! It should return FALSE when it doesn't, TRUE when it does.

  #True if active_conflict exists and False if it does not
  if os.path.exists("%s/conflict_data/active_conflict/" % (env.fabflee_root)):
    print("test_clear_active_conflict: True")
    return True
  else:
    print("test_clear_active_conflict: False")
    return False
