import os
import urllib.request
from flask import Flask, flash, request, redirect, render_template, url_for, json
from werkzeug.utils import secure_filename
from flask import Flask

import librosa
import numpy as np
from joblib import load
import os
from sklearn.preprocessing import StandardScaler
from pydub import AudioSegment

from os.path import join
import random
import json
import pickle
from collections import Counter

ALLOWED_EXTENSIONS = ["wav", "mp3", "ogg", "flv", "wma", "aac", "mp4"]
UPLOAD_FOLDER = os.getcwd() + '/uploads'

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hard to guess string'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# maximum file size
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

def allowed_file(filename):
    fn_parts = filename.split(".")
    extension = fn_parts[-1]
    return extension in ALLOWED_EXTENSIONS

@app.route('/')
def upload_form():
    path = os.getcwd() + '/uploads'
    songs = os.listdir(path)
    modelspath = os.getcwd() + '/static/models'
    models = os.listdir(modelspath)
    return render_template('upload.html', songs=songs, models=models)


@app.route('/', methods=['POST', 'GET'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('File(s) successfully uploaded')
            return redirect(url_for('.result', path=filename))

@app.route('/result')
def result():
    path = request.args['path']
    result, highest = classify(path)
    return render_template("result.html", path=path, result=result, highest=highest)


def classify(path):
    fullpath = os.getcwd() + '/uploads/' + path
    fn_parts = path.split(".")
    song_name = fn_parts[0]
    extension = fn_parts[-1]

    sound = None
    if extension == 'mp3':
        sound = AudioSegment.from_mp3(fullpath)
    
    if extension == 'ogg':
        sound = AudioSegment.from_ogg(fullpath)
    
    if extension == 'flv':
        sound = AudioSegment.from_flv(fullpath)
    
    if extension == "mp4":
        sound = AudioSegment.from_file(fullpath, "mp4")

    if extension == "wma":
        sound = AudioSegment.from_file(fullpath, "wma")
    
    if extension == "aac":
        sound = AudioSegment.from_file(fullpath, "aac")

    song = "song.wav"
    sound.export(song, format="wav")

    model = 'finalized_model.sav'
    MODEL = os.getcwd() + '/static/models/' + model

    y, sr = librosa.load(song)
    t = int(librosa.get_duration(y=y, sr=sr))
    if (t < 20):
        return redirect(request.url)

    collection = []
    for i in range(0, 10):
        random_time = random.randint(0, t-3) 
        y1, sr1 = librosa.load(song, mono=True, duration=2, offset=random_time)
        ps = librosa.feature.melspectrogram(y=y1, sr=sr1, hop_length = 256, n_fft = 512, n_mels=64)
        collection.append(librosa.power_to_db(ps**2))
        '''
        print(start, dur)
        y, sr = librosa.load(song, offset=start, duration=dur)
        start += dur
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        rmse = librosa.feature.rms(y=y)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        features = [np.mean(chroma_stft), np.mean(rmse), np.mean(spec_cent), np.mean(spec_bw), np.mean(rolloff),
                    np.mean(zcr)]
        for e in mfcc:
            features.append(np.mean(e))
        collection.append(features)
        '''
    
    '''
    scaler = StandardScaler()
    collection = scaler.fit_transform(np.array(collection, dtype=float))
    '''
    collection = np.array([x.reshape((64, 173, 1)) for x in collection])

    #classifier = load(SVM_MODEL)
    with open(MODEL, "rb") as file:
        loaded_model = pickle.load(file)

    result = loaded_model.predict_classes(collection)

    print(result)

    genres = {0 : "blues", 1: 'classical', 2: 'country', 3: 'disco', 4: 'hiphop', 5: 'jazz', 6: 'metal', 7: 'pop' , 8: 'reggae', 9: 'rock'}

    percentages = ["%.2f"%(float(result.tolist().count(x)) / float(len(genres)) * 100) for x in genres.keys()]
    # print the genre percentage
    return percentages, genres[percentages.index(max(percentages))]

if __name__ == "__main__":
    app.jinja_env.auto_reload = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(host='0.0.0.0', port="8080", debug=True, threaded=False)
