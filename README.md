## *Hydrograph peak-shaving using a graph-theoretic algorithm for placement of hydraulic control structures*

Preprint available at: https://arxiv.org/abs/1809.03838

Code
----

- /notebooks/partition.py: contains the controller placement algorithm.
- /notebooks/swmm.py: contains the code used to generate SWMM models of the drainage network.
- /notebooks/run_simulations.py: runs all simulations.

Notebooks
---------

- /notebooks/FIG_elev_and_river.ipynb: code used to generate Fig. 1
- /notebooks/FIG_weights.ipynb: code used to generate Figs. 2, S1, S2
- /notebooks/FIG_acc_and_wacc.ipynb: code used to generate Fig. 3
- /notebooks/FIG_partition.ipynb: code used to generate Figs. 4, S3
- /notebooks/FIG_performance_alt.ipynb: code used to generate Fig. 5
- /notebooks/FIG_num_controllers.ipynb: code used to generate Figs. 6, S8
- /notebooks/FIG_placement_view.ipynb: code used to generate Fig. 7
- /notebooks/FIG_performance.ipynb: code used to generate Figs. S4, S5
- /notebooks/FIG_full_performance.ipynb: code used to generate Figs. S6, S7
- /notebooks/generate_swmm_files.ipynb: code used to generate SWMM simulations
- /notebooks/placement_experiments_50pct_phi10.ipynb: code used to generate placement experiments
- /notebooks/find_channels.ipynb: code used to compare thesholded channels to those found in NHD dataset

Data
----

Download the following data into the /data directory:
- /inp: https://s3.us-east-2.amazonaws.com/controller-placement-data/inp.zip
  - Contains input files for the SWMM model
- /out: https://s3.us-east-2.amazonaws.com/controller-placement-data/out.zip
  - Contains output files from the SWMM model
- /n30w100_con: https://s3.us-east-2.amazonaws.com/controller-placement-data/n30w100_con.zip
  - DEM data
- /n30w100_dir: https://s3.us-east-2.amazonaws.com/controller-placement-data/n30w100_dir.zip
  - Flow direction data
