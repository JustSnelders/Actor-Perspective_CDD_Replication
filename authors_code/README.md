# Multi-Perspective Concept Drift Detection on BPIC17 Event Knowledge Graph
---------------------

This repository extends the [Event Knowledge Graph for BPIC17 Data repository](https://github.com/PromG-dev/ekg_bpic17)
with a module for multi-perspective concept drift detection. This module reproduces the experiments and results provided
in the corresponding paper.

## Installation

### Neo4j

The code assumes that Neo4j is installed.

Install [Neo4j](https://neo4j.com/download/):

- Use the [Neo4j Desktop](https://neo4j.com/download-center/#desktop)  (recommended), or
- [Neo4j Community Server](https://neo4j.com/download-center/#community)

### PromG

PromG should be installed as a Python package using pip
`pip install promg=1.0.10`.

The source code for PromG can be found [PromG Core Github repository](https://github.com/PromG-dev/promg-core).

---------------------

## Get started

### <a name="create_db"></a> Create a new graph database

- Create a New Graph Data In Neo4j Desktop
    1. Select `+Add` (Top right corner)
    2. Choose Local DBMS or Remote Connection
    3. Follow the prompted steps (the default password we assume is 12345678)
- Install APOC (see https://neo4j.com/labs/apoc/)
    - Install `Neo4j APOC Core library`:
        1. Select the database in Neo4j desktop
        2. On the right, click on the `plugins` tab > Open the `APOC` section > Click the `install` button
        3. Wait until a green check mark shows up next to `APOC` - that means it's good to go!
    - Install `Neo4j APOC Extended library`
        1. Download the [appropriate release](https://github.com/neo4j-contrib/neo4j-apoc-procedures/releases) (same
           version numbers as your Neo4j version)
            1. Look for the release that matches the version number of your Neo4j Database.
            2. Download the file `apoc-[your neo4j version]-extended.jar`
        2. Locate the `plugins` folder of your database:  
           Select the Neo4j Server in Neo4j Desktop > Click the three dots > Select `Open Folder` > Select `Plugins`
        4. Put `apoc-[your neo4j version]-extended.jar` into the `plugins` folder of your database
        5. Restart the server (database)
    - Configure extra settings using the configuration file `$NEO4J_HOME/conf/apoc.conf`
        1. Locate the `conf` folder of your database  
           Select the Neo4j Server in Neo4j Desktop > Click the three dots > Select `Open Folder` > Select `Conf`
        2. Create the file `apoc.conf`
        3. Add the following line to `apoc.conf`: `apoc.import.file.enabled=true`.
- Ensure to allocate enough memory to your database, advised: `dbms.memory.heap.max_size=10G`
    1. Select the Neo4j Server in Neo4j Desktop > Click the three dots > Select `Settings`
    2. Locate `dbms.memory.heap.max_size=512m`
    3. Change `512m` to `10G`

### Set configuration file

- Configuration; `config.yaml`
    - Set the URI in `config.yaml` to the URI of your server. Default value is `bolt://localhost:7687`.
    - Set the password in `config.yaml` to the password of your server. Default value is `12345678`.
    - Set the import directory in `config.yaml` to the import directory of your Neo4j server. You can determine the
      import directory as follows:
        1. Select the Neo4j Server in Neo4j Desktop > Click the three dots > Select `Open Folder` > Select `Import`
        2. This opens the import directory, so now you can copy the directory.

### Store Data

We provide data and scripts for BPI Challenge 2017; store the original data in CSV format in the directory `/data`.
The datasets are available from:

            Esser, Stefan, & Fahland, Dirk. (2020). Event Data and Queries
            for Multi-Dimensional Event Data in the Neo4j Graph Database
            (Version 1.0) [Data set]. Zenodo. 
            http://doi.org/10.5281/zenodo.3865222

### Install PromG Library for Script

The library can be installed in Python using pip: `pip install promg==1.0.10`.
The source code for PromG can be found [PromG Core Github repository](https://github.com/PromG-dev/promg-core).

---------------------

## Provided Scripts

### Configuration files

- `config.yaml` Configuration of the database.
- `config_analysis.yaml` Configuration of the clustering (number of clusters, filter for variants to consider in the
  clustering) and concept drift analysis (penalty, window size, features to use).

### Main script

There is one script that implements all queries for constructing, manipulating and analyzing the Event knowledge graph :
`main.py`

The script implements the following steps:

0. `step_clear_db` Clears the Event knowledge graph.
1. `step_populate_graph` Imports a normalized event table of BPIC17 from CSV files and executes several data modeling
   queries to
   construct an event knowledge graph using the semantic header.
   _See [Event Knowledge Graph for BPIC17 Data repository](https://github.com/PromG-dev/ekg_bpic17) for more information
   on the semantic header._
2. `step_build_tasks` Identifies and constructs task instances. Requires step 1.
3. `step_cluster_tasks` Clusters and aggregates task instances based on the clustering result. Requires steps 1-2.
4. `step_visualize_clusters` Visualizes the task instance variants grouped per cluster as aid for inspecting the result
   of the task clustering. Requires steps 1-3.
5. `step_process_level_drift_detection` Detects change points in the entire process based on a set of specified
   features (see Configuration files). Requires steps 1-3.
6. `step_actor_drift_detection` Detects change points on the level of individual actors based on a set of specified
   features (see Configuration files). Requires steps 1-3.
7. `step_collab_drift_detection` Detects change points on the level of actor interactions perspective (see Configuration
   files). Requires steps 1-3.
8. `step_compare_to_process_drift` Compares detected change points in specific actors or collabs (set
   in `config_analysis.yaml`) to detected process level change points and outputs results in text file. Requires steps
   1-3 and 5-7.
9. `step_calculate_magnitude_signal_changes` Calculates the magnitude of the signal changes for each feature in the
   specified feature set for each detected process level change point. Requires steps
   1-3 and 5.
10. `step_calculate_overall_average_max_signal_change` Calculates the average and maximum signal change across all
    change points and all features for each feature set. Requires steps 1-3, 5 and 8.
11. `step_detailed_change_signal_analysis` Outputs a detailed comparison of task/task variant features vs control-flow-based features, see Table 2 in corresponding paper. Requires step 1-3, 5, 9 and 10.

All steps specified in `main.py` can be switched on/off. Please note that most steps assume graph constructs or other
results from preceding steps.

------------------------
## How to use

1. Set the configuration in `config.yaml`.
    - For database settings, see [Create a new graph database](#create_db).
    - Make sure the set password matches that of the Neo4j server.
2. Set the configuration in `config_analysis.yaml`.
3. Turn desired steps on/off in `main.py`.
4. Start the Neo4j server.
5. Run `main.py`.
