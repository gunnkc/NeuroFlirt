import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler

from scores_enum import Score, Engagement, Attraction, Satisfaction

def parse_metrics(raw: list[list]) -> tuple[list, list, list]:
    score = []
    arousal = []
    valence = []
    boring = []

    for inner_list in raw:
        valence.append(inner_list[0])
        inner_list[1] = 10 - inner_list[1]
        boring.append(inner_list[1])
        arousal.append(inner_list[2])

        score.append(sum(inner_list) / 4)
    
    return (score, arousal, valence, boring)

def get_status(int_list: list[int], label: str) -> str:
    avg_value = round(np.mean(int_list))
    
    if label == "Score":
        return Score(avg_value).name   
    elif label == "Engagement":
        return Engagement(avg_value).name
    elif label == "Attraction":
        return Attraction(avg_value).name
    elif label == "Satisfaction":
        return Satisfaction(avg_value).name
    else:
        raise ValueError("Invalid label")
    
# the most basic training data that includes features from full windows after processing
# with batch size input
def generate_summary_stats_training_data(df, batch_size):
    # df : unprocessed dataframe (No batch processing)
    # batch_size : indicate processing size (# of seconds)
    # training columns without creating covariance matrices
    train_cols = ['mean_tp9', 'mean_tp10', 'mean_af7', 'mean_af8',
                     'std_tp9', 'std_tp10', 'std_af7', 'std_af8',
                     'vari_tp9', 'vari_tp10', 'vari_af7', 'vari_af8',
                     'max_tp9', 'max_tp10', 'max_af7', 'max_af8',
                     'min_tp9', 'min_tp10', 'min_af7', 'min_af8',
                     'skew_tp9', 'skew_tp10', 'skew_af7', 'skew_af8',
                     'kurt_tp9', 'kurt_tp10', 'kurt_af7', 'kurt_af8']
    train_cols_dict = dict(zip(train_cols, [[] for i in range(len(train_cols))]))

   
   
    for start in range(0, df.shape[0], batch_size):
        end = start + batch_size
        if end > len(df):
            end = len(df)
        batch = df[start:end]
       
        # add mean summary stats on all 4 channels
        train_cols_dict['mean_tp9'].append(batch['TP9'].mean())
        train_cols_dict['mean_tp10'].append(batch['TP10'].mean())
        train_cols_dict['mean_af7'].append(batch['AF7'].mean())
        train_cols_dict['mean_af8'].append(batch['AF8'].mean())
       
        # add std summary stats on all 4 channels
        train_cols_dict['std_tp9'].append(np.std(batch['TP9'], ddof=1))
        train_cols_dict['std_tp10'].append(np.std(batch['TP10'], ddof=1))
        train_cols_dict['std_af7'].append(np.std(batch['AF7'], ddof=1))
        train_cols_dict['std_af8'].append(np.std(batch['AF8'], ddof=1))
       
        # add variance
        train_cols_dict['vari_tp9'].append(np.var(batch['TP9'], ddof=1))
        train_cols_dict['vari_tp10'].append(np.var(batch['TP10'], ddof=1))
        train_cols_dict['vari_af7'].append(np.var(batch['AF7'], ddof=1))
        train_cols_dict['vari_af8'].append(np.var(batch['AF8'], ddof=1))
       
        # add min
        train_cols_dict['min_tp9'].append(batch['TP9'].min())
        train_cols_dict['min_tp10'].append(batch['TP10'].min())
        train_cols_dict['min_af7'].append(batch['AF7'].min())
        train_cols_dict['min_af8'].append(batch['AF8'].min())
       
        # add max
        train_cols_dict['max_tp9'].append(batch['TP9'].max())
        train_cols_dict['max_tp10'].append(batch['TP10'].max())
        train_cols_dict['max_af7'].append(batch['AF7'].max())
        train_cols_dict['max_af8'].append(batch['AF8'].max())
       
        # add skew
        train_cols_dict['skew_tp9'].append(skew(np.array(batch['TP9'])))
        train_cols_dict['skew_tp10'].append(skew(np.array(batch['TP10'])))
        train_cols_dict['skew_af7'].append(skew(np.array(batch['AF7'])))
        train_cols_dict['skew_af8'].append(skew(np.array(batch['AF8'])))
       
        # add kurtosis
        train_cols_dict['kurt_tp9'].append(kurtosis(np.array(batch['TP9'])))
        train_cols_dict['kurt_tp10'].append(kurtosis(np.array(batch['TP10'])))
        train_cols_dict['kurt_af7'].append(kurtosis(np.array(batch['AF7'])))
        train_cols_dict['kurt_af8'].append(kurtosis(np.array(batch['AF8'])))
   
    output_training_df = pd.DataFrame(train_cols_dict)

    scaler = StandardScaler()
    output_training_df = scaler.fit_transform(output_training_df)

    return output_training_df