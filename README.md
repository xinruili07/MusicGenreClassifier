# MusicGenreClassifierWebsite

Introduction
------------
Building a music genre classification model for the McGill AI Bootcamp. Using the GTZAN Genre Collection dataset consisting of 1000 audio tracks each 30 seconds long. It contains 10 genres, each represented by 100 tracks. The tracks are all 22050Hz Mono 16-bit audio files in .wav format. The dataset consists of 10 genres: Blues, Classical, Country, Disco, Hip-hop, Jazz, Metal, Pop, Reggae, Rock. Each genre contains 100 audio tracks. This is the Web Application that will be used to demonstrate my project.
 

Requirements
------------
* Flask (1.1.1)
* Werkzeug (0.16.0)
* Numpy (1.12.1)
* Librosa (0.7.1)
* Pydub (0.18.0)
* ffmpeg (4.2.1)

Installation
-------------
* `git clone https://github.com/xinruili07/MusicGenreClassifierWebsite.git`
* `pip install -r requirements.txt`
* `brew install ffmpeg~
* `python3 app.py`
* *App will run on* `localhost:8080`

## Music Genre Classifier App
The web application is written in Python using Flask. It uses a saved model (more details can be found at https://github.com/xinruili07/MusicGenreClassifier)for finding the genre of input song. 

## Results
With 10 genre classes, we are getting an test accuracy of 77%



