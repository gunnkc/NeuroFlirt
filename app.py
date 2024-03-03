from time import time, sleep

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from rizz_calculator import RizzCalculator
from utils import parse_metrics, get_status

RELOAD_SECS = 2
NUM_TO_STORE = 10

CSS = """
<div style='display: flex; align-items: center; justify-content: center; height: 100%; padding-top: 100px;'>
    <p style='text-align: center;'> {} </p>
</div>
"""

def refresh_main():
    ensure_model_initialized()
    model = st.session_state['model']

    st.title("NeuroFlirt")
    st.markdown('By Ethan K, Gunn C, Michael P, Nathan C')

    # Placeholder for the metrics display
    metrics_placeholder = st.empty()

    # Check if it's time to update
    if 'last_update' not in st.session_state or time() - st.session_state['last_update'] > RELOAD_SECS:
        st.write("Acqurining data -- May take up to 30 seconds...")
        metrics = model.get_metrics()
        print(metrics)
        st.session_state['last_update'] = time()  # Update the last update time

        # Display metrics in the placeholder
        with metrics_placeholder.container():
            display_metrics(metrics)
    else:
        # If not time to update, show a message or you can choose to do nothing
        with metrics_placeholder.container():
            st.write("Waiting for the next update...")

    # Optional: Add a manual refresh button
    if st.button("Refresh Now"):
        st.session_state['last_update'] = 0  # Force an update on next rerun
    
    if st.button("Stop Connection"):
        st.write("Closing Connection")
        model.stop_predictions()
        st.stop()


def simple_app():
    ensure_model_initialized()
    model = st.session_state['model']

    st.title("ElectroFlirt")
    st.markdown('By Ethan K, Gunn C, Michael P, Nathan C')

    sleep(10)
    metrics = model.get_metrics()

    print(len(metrics))

    display_metrics(metrics)

def display_metrics(raw_metrics):
    print("Rendering metrics...")
    st.title("Statistics")
    if not raw_metrics:
        return

    score, attraction, satisfaction, engagement = parse_metrics(raw_metrics)

    metrics = [
        {"title": "**Score**", "data": score, 
         "status": "How is the conversation going?: \n" + get_status(score, "Score")},
        {"title": "**Engagement**", "data": engagement, 
         "status": "How engaged is the other person?: \n" + get_status(engagement, "Engagement")},
        {"title": "**Attraction**", "data": attraction, 
         "status": "How much is the other person attracted?: \n" + get_status(attraction, "Attraction")},
        {"title": "**Satisfaction**", "data": satisfaction, 
         "status": "How satisfied is the other person?: \n" + get_status(satisfaction, "Satisfaction")}
    ]
    
    for metric in metrics:
        title = metric['title']
        data = metric['data']
        text = metric['status']

        col1, col2 = st.columns([5, 2])

        with col1:
            st.write(title)
            fig, ax = plt.subplots(figsize=(2.5, 1))
            x = range(-1 * len(data), 0)
            y = data
            ax.plot(x, y, marker='o', linestyle='-', label=title, color='pink')
            ax.tick_params(axis='both', which='major', labelsize=4)
            ax.set_ylim(0, 10)
            plt.tight_layout()
            # fig.patch.set_facecolor('tab:gray')
            # ax.set_facecolor('tab:gray')
            st.pyplot(fig, use_container_width=False)
            print("plot complete")
        
        with col2:
            st.markdown(CSS.format(text), unsafe_allow_html=True)


def ensure_model_initialized():
    if 'model' not in st.session_state:
        st.session_state['model'] = RizzCalculator()
        st.session_state['model'].start_predictions()


if __name__ == "__main__":
    refresh_main()
