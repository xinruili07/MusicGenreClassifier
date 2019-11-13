from flask import Flask, url_for, redirect, render_template, send_file, request, flash
from flask_bootstrap import Bootstrap
from flask_moment import Moment
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired

from os.path import join
from sklearn.externals import joblib
import random
import json
import librosa
import os
import pickle

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hard to guess string'

bootstrap = Bootstrap(app)
moment = Moment(app)

spectogram_data = []
song_name = ""
genres = {'blues': 0, 'classical': 1, 'country': 2, 'disco': 3, 'hiphop': 4, 'jazz': 5, 'metal': 6, 'pop': 7, 'reggae': 8, 'rock': 9}

curr_dir = os.path.dirname(__file__)
filename = 'genre_classifier_v1.pickle'

with open (join(curr_dir, "models/genre_classifier_v1" + ".pickle"), "rb") as file:
	model = pickle.load(file)

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500


@app.route ("/", methods=["POST", "GET"])
def main_page ():
    allowed_extensions = ["wav", "mp3", "ogg", "flac", "wma", "aac"]
    if request.method == "POST":
        if "sound_file_upload" not in request.files:
            flash("Error in uploading music file")
            return redirect(request.url)

        file = request.files["sound_file_upload"]
        if file.filename == "":
            flash("No file has been selected")
            return redirect(request.url)

        fn_parts = file.filename.split (".")
        global song_name
        song_name = fn_parts[0]
        extension = fn_parts[-1]

        if extension not in allowed_extensions:
            flash("This file format is not supported")
            return redirect(request.url)

        global spectogram_data
        
        duration = librosa.get_duration(file)
        if (duration < 10):
            return redirect(request.url)

        for index in range(5):
            random_time = random.randint(0,duration-3) 
            y, sr = librosa.load(file, mono=True, duration=2, offset=random_time)
            ps = librosa.feature.melspectrogram(y=y, sr=sr, hop_length = 256, n_fft = 512, n_mels=128)
            spectogram_data.append(librosa.power_to_db(ps**2))

        return render_template("index.html", file_uploaded = True)
    return render_template("index.html", file_uploaded = False)

@app.route ("/result", methods=["POST"])
def analysis ():
    global spectogram_data
    if spectogram_data is None:
        return "", 400

    y_pred = model.predict_classes(spectogram_data)
    most_common = max(set(y_pred), key=y_pred.count)
    genre = genres[most_common]
    result_dict = {"genre": genre}
    return json.dumps(result_dict), 200
    return render_template('results.html',songname=song_name genre=genre)
if __name__ == '__main__':
    app.run(debug=True)