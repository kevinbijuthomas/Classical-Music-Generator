# Classical-Music-Generator

This repository contains the code for a website that allows you to generate and download melodies (as MIDI files) created by a Long Short-Term Memory (LSTM) machine learning model trained on classical music. 

I had deployed the model <a href="https://classicalmusic-35dwnoodbq-uc.a.run.app">here</a> on Google Cloud Run using Docker to containerize the application. I also added a trigger that watches this repository to maintain CI/CD. 

However, I cannot afford to pay for constant GPU backing and so the service is currently unavailable.

To run the website on your local machine:
1. Clone the repository.
2. Make sure to install all the libraries shown in <i>requirements.txt</i>.
3. Run <b><i>train.py</i></b> to create the model "200Epochs.h5". You will need GPU support to train the model.
4. Run <b><i>main.py</i></b> to launch the flask application on your local machine.

An issue with the model is that it does not account for duration of notes and offsets between notes, and so I incorporated a uniform duration and offset throughout. Though this is not ideal, the model still creates songs that are of decent quality (in my opinion).
