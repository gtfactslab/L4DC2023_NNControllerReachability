# Interval Reachability of Nonlinear Dynamical Systems with Neural Network Controllers

This repository is associated with the manuscript "Interval Reachability of Nonlinear Dynamical Systems with Neural Network Controllers" submitted to L4DC 2023. 

## Final Manuscript

The folder "Final_Manuscript_with_Proofs" contains the final version of our paper with the proof of all the results. 

## Code

To recreate the figures from the paper, run the following script
```
python experiments.py --model MODEL --runtime_N 10
```
replacing `MODEL` with `vehicle` or `quadrotor`.

More information about how to properly setup a conda environment to interface correctly with the included submodules is coming soon.

<!-- ### Submodules
Note: the submodule `nn_robustness_analysis` is used for comparison with  -->
