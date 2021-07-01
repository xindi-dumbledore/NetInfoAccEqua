# NetInfoAccEqua
 Codes for "Information Access Equality on Network Generative Models"

## Folders
### codes
1. `exp_model.py`: experiments on comparing different models
2. `exp_param.py`: experiments on comparing different parameters
3. `exp_real.py`: experiments on real networks
4. `exp_results_draw.py`: generate figures from experiment results
5. `fairness_meaure.py`: fairness measure calculation
6. `network_models.py`: functions to generate network models
7. `simulation_param.py`: parameter setting for the simulation

### exp_results
Store experiment results from `exp_model.py`, `exp_param.py` and `exp_real.py`. Currently only provided a folder structure.

### figures
Store figures from running `exp_results_draw.py`.

### datasets
The datasets is from the paper "Homophily and minority-group size explain perception biases in social networks" by Lee et. al. Their github repository link is [NtwPerceptionBias](https://github.com/frbkrm/NtwPerceptionBias).

## Cookbook
1. run `exp_model.py`, `exp_param.py` and `exp_real.py`
2. run `exp_results_draw.py`.
3. One can change the simulation parameter in `simulation_param.py`.

