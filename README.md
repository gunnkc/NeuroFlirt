# NeuroFlirt: Re-Imagining the Online Dating Industry

## Overview

NeuroFlirt introduces a groundbreaking approach to online dating by leveraging brain activity to ensure deep compatibility between users. By focusing on brain activity-informed scores, we streamline the search process, enhance match quality, and reduce overwhelm in the dating app landscape. Our solution addresses the divide in online dating experiences, offering a scientifically informed pathway to meaningful connections.\
\
[Link to Presentation](https://docs.google.com/presentation/d/1Erc2AedQndgK8YLnwfZG6LBhWpSU4yxb59X0fC0wA8Y/edit?usp=sharing)

## Features

- **Focused Interactions:** Matches are based on brain activity, streamlining searches and saving time.
- **Enhanced Match Quality:** A brain activity-based scoring system guarantees deep compatibility.
- **Reduced Overwhelm:** Limits choices by focusing on quality, simplifying decisions.
- **Live Emotional Feedback:** Provides real-time tracking of emotions such as attraction and happiness during conversations.

## Training Data

- **Subjects:** 28 total subjects participating in 4 different games.
- **Data:** ~4,000,000 rows of brain activity data processed using Muse with 4 electrode channels.

## Machine Learning Model

- **Data Processing:** Utilizes regional averaging for clarity from 14 electrode channels.
- **Model:** LightGBM with predictions made every 30 seconds and a regression model achieving ~2 RMSE on a scale of 0 - 8.

## Ethical Considerations

We prioritize privacy, consent, data security, and algorithmic fairness to ensure a safe and equitable user experience.

## Other Applications

NeuroFlirt's technology also has potential applications in healthcare, education, market research, and couple therapy.

## Getting Started

Clone the repo, then use the following code to setup a virtual environment and install all dependencies.
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
\
The application requires a connection to [MindMonitor](https://mind-monitor.com/). It is possible to find alternatives but it will require adjustments to the code. Once a connection is set up and data is streaming to the local device, you can launch the app locally using the following command.
```bash
python3 -m streamlit app.py
```

## Contact

Created for the purpose of Neuroengineering Hackathon, March 3, 2024.\
[Gunn Chun](mailto:gunncre@gmail.com)\
[Nathan Chen](mailto:nchen35@uw.edu)\
[Ethan Kawahara](mailto:ekawah@uw.edu)\
[Michael Petta](mailto:mpetta@uw.edu)
