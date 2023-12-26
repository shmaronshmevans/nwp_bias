# nwp_bias
This will be a repository for learning the forecast error and bias for the nwp models: GFS, NAMM, HRRR.
I utilize a Long-Short-Term-Memory ML model utilizing pytorch to learn and predict the 2-meter-temperature error of NWP models. This readme will serve as a way to download, clean and utilize the data for similar applications. Either to recreate and test my own work, or to utilize with your own Mesonet. All I ask is that you reference our work when you do. 

## Downloading Data
For downloading the GFS, NAM and HRRR files, please read and follow the instructions of my co-athor. Lauriana Gaudet's github repo can be found: https://github.com/lgaudet/AI2ES-notebooks

Use these notebooks:
- s3_download.ipynb
- get_resampled_nysm_data.ipynb
- cleaning_bible-NYS-subset.ipynb
- all_models_comparison_to_NYSM.ipynb

Downloading NYSM data can be found here: https://www.nysmesonet.org/weather/requestdata

## Cleaning Data 
### src
| notebook | description |
|-----------|------------|
|optimizer.config| config dictionary for comet hyperparameter tuning|
|switchboard.py| init script for fsdp training of lstm|
|t2m_hrrr_fh2_western_plateau_opt.py| comet hyperparameter training script|


### bias 
| notebook | description |
|-----------|------------|
|all_models_comparison_to_mesos_lstm.py| cleans hrrr data that was downloaded from lgaudet's github repo and only keeps variables and locations for nysm|
|forecase_hr_parquet_builder.py| reads in hrrr data by init time and compiles into temporally linear parquet by valid_time for an input forecast hour |

 ## LSTM
 ### data
| notebook | description |
|-----------|------------|
|create_data_for_lstm.py| compiles data from hrrr + nysm into a dataframe, indexed by valid_time. Then normalizes data. Then seperates into training and test sets. |
|hrrr_data.py| compiles cleaned data from hrrr into dataframe for lstm |
|nysm_data.py|compiles cleaned data from nysm into dataframe for lstm |

### evaluate
| notebook | description |
|-----------|------------|
|eval_single_gpu.py|evaluate model performance on a single gpu|
|fsdp.py|train an lstm with fully-shared-data-paralellization on multi gpus|
|lstm_single_gpu.py|train lstm on single gpu (recommended)|
|retrain_lstm.py|used to train lstm from a previously trained model|

### notebooks
| notebook | description |
|-----------|------------|
| emd.ipynb | plot empirical mode decomposition of meteorological variables|
| plot_fh_drift.ipynb | plot the difference in loss by different forecast hours for different lstm models | 
| visualie_fsdp_lstm_output.ipynb | plots of loss by meteorological variable |

