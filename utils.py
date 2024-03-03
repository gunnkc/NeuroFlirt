import numpy as np

from scores_enum import Score, Engagement, Attraction, Satisfaction

def parse_metrics(raw: list[list]) -> tuple[list, list, list]:
    score = []
    arousal = []
    valence = []
    boring = []

    for inner_list in raw:
        arousal.append(inner_list[0])
        valence.append(inner_list[1])
        boring.append(10 - inner_list[2])
        score.append(sum(inner_list) / 3)
    
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