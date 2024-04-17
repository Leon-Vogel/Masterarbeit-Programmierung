# ================================
# THIS FILE HAS BEEN AUTOMATICALLY GENERATED. DO NOT OVERWRITE IT MANUALLY
# WHEN YOU CREATE NEW SUBFOLDERS OR WHEN YOU WANT TO MANIPULATE THIS FILE,
# GO TO src/executable/. AND OVERWRITE THE MASTER FILE _dir_init.py.
# THEN RUN THIS MASTER SCRIPT. IT AUTOMATICALLY CREATES A _dir_init.py
# FILE IN EVERY PROJECT SUBFOLDER AND SETS THE CURR_DIR_LEVEL
CURR_DIR_LEVEL = 2 # 0 = root directory src/executable/.
# ================================

# init root dir
import sys
import os
FILE_DIR = (
    os.getcwd()
    if "DATABRICKS_RUNTIME_VERSION" in os.environ
    else os.path.dirname(os.path.abspath(__file__))
)
PROJECT_ROOT_DIR = os.path.abspath(
    os.path.join(os.path.abspath(FILE_DIR), *([".."] * CURR_DIR_LEVEL))
)
sys.path.append(PROJECT_ROOT_DIR)

# add additional imports
import path_definitions
import path_definitions_XAI

# add additional parameters
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


