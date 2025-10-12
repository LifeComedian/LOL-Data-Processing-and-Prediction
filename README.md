# LOL-Data-Processing-and-Prediction

dataset link: https://www.kaggle.com/datasets/weitianxue/lol-matches-timeline-dataset

**Main programs**
1. Data Acquisition

get_players.py
Get the top1000 ranked players in KR region as the seeds for dataset collection.

LOL_dataset_collection.ipynb
Collects match data from Riot’s MATCH-V5 API using player seeds, downloads timeline JSONs and participant stats into raw.csv, and includes pagination and retry logic for reproducibility.

2. Data Processing

data_processing.py
Converts timeline JSONs into per-minute CSVs, aligns them with blue-win labels and optional champion metadata, aggregates match-level features, and produces train/validation splits for tabular model training.

get_trajectory.ipynb
Generates cumulative movement trajectory images per participant and minute, saving them in structured folders for multimodal training.

3. Model Training

train_CNN.py
Trains a fusion model combining tabular and CNN-encoded image data, with checkpointing, AMP support, and resume functionality.

train_tabular.py
Implements a lightweight MLP using only tabular features, serving as a baseline and ablation comparison.

train_advance.py
Improve the CNN model with the temporal transformer.

4. Evaluation

eval_interval_CNN.py
Evaluates trained models across different time intervals in both tabular and fusion modes using CNN, reporting accuracy and AUC for performance comparison.

eval_interval_advance.py
Evaluates trained models across different time intervals using the advanced temporal model.
