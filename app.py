from time import sleep

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from rizz_calculator import RizzCalculator
from utils import parse_metrics, get_status

RELOAD_SECS = 1
NUM_TO_STORE = 10

model = RizzCalculator()
initial = True

def main():
    global model
    global initial

    st.title("ElectroFlirt")
    st.markdown('By Ethan K, Gunn C, Michael P, Nathan C')

    if 'data' not in st.session_state:
        if initial:
            st.write('Retrieving data -- May take up to 30 seconds')
            sleep(15)
            initial = False
        st.session_state['metrics'] = model.get_metrics()
    
    else:
        st.session_state['metrics'].extend(model.get_metrics())
    
    if len(st.session_state['metrics']) < 10:
        sleep(10)  # Arbitrary
        st.experimental_rerun()
    
    st.session_state['metrics'] = st.session_state['metrics'][-10:]

    display_metrics(st.session_state['metrics'])

    sleep(5)
    st.experimental_rerun()

def display_metrics(raw_metrics):
    st.title("Statistics")

    score, attraction, satisfaction, engagement = parse_metrics(raw_metrics)

    metrics = [
        {"title": "Score", "data": score, 
         "status": "How is the conversation going?: \n\n" + get_status(score, "Score")},
        {"title": "Engagement", "data": engagement, 
         "status": "How engaged is the other person?: \n\n" + get_status(engagement, "Engagement")},
        {"title": "Attraction", "data": attraction, 
         "status": "How much is the other person attracted?: \n\n" + get_status(attraction, "Attraction")},
        {"title": "Satisfaction", "data": satisfaction, 
         "status": "How satisfied is the other person?: \n\n" + get_status(satisfaction, "Satisfaction")}
    ]
    
    for metric in metrics:
        title = metric['title']
        data = metric['data']
        text = metric['text']

        col1, col2 = st.columns([5, 1])

        with col1:
            st.write(title)
            fig, ax = plt.subplots(figsize=(3, 1))
            x = range(-10, 1)
            y = data
            ax.plot(x, y, marker='o', linestyle='-', label=title)
            ax.tick_params(axis='both', which='major', labelsize=4)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=False)
        
        with col2:
            st.write(text)


if __name__ == "__main__":
    main()
