# import dependencies
import os
from flask import render_template, redirect, url_for, request, flash
from app import app
from werkzeug.utils import secure_filename
# import the function for predictions
from app.main.test import predict_bird, predict_pets, predict_tumors 


# utility code
UPLOAD_FOLDER = r"app/uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])
def allowed_file(filename): 
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# routes
@app.route('/home')
@app.route('/')
def index():
    return render_template('home.html')

@app.route('/birds-species', methods=['GET', 'POST'])
def model_birds():
    return render_template('birds.html')

@app.route('/birds-result', methods=['GET', 'POST'])
def upload_birds():
    if request.method == 'POST':
        #check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        f = request.files['file']
        
        # if user does not select file, browser also submit a empty part without filename
        if f.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if f and allowed_file(f.filename):
            filename = secure_filename(f.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            f.save(file_path.replace("\\","/"))
        prediction = predict_bird(file_path)
        # replace sumn
        start = ".."
        file_path = start + file_path[3:]
        file_path = file_path.replace("\\","/")
        #return file_path
        return render_template('birds-result.html', prediction=prediction, file_path=file_path)
    return " "


@app.route('/cats-dogs', methods=['GET', 'POST'])
def model_pets():
    return render_template('pets.html')

@app.route('/pets-result', methods=['GET', 'POST'])
def upload_pets():
    if request.method == 'POST':
        #check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        f = request.files['file']
        # if user does not select file, browser also submit a empty part without filename
        if f.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if f and allowed_file(f.filename):
            filename = secure_filename(f.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            f.save(file_path.replace("\\","/"))
        prediction = predict_pets(file_path)

        # replace sumn
        start = ".."
        file_path = start + file_path[3:]
        file_path = file_path.replace("\\","/")
        #return file_path
        return render_template('pets-result.html', prediction=prediction, file_path=file_path)
    return " "

@app.route('/tumors', methods=['GET', 'POST'])
def model_tumors():
    return render_template('tumors.html')

@app.route('/tumors-result', methods=['GET', 'POST'])
def upload_tumors():
    if request.method == 'POST':
    
    #check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
            
        f = request.files['file']

    # if user does not select file, browser also submit a empty part without filename
        if f.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if f and allowed_file(f.filename):
            filename = secure_filename(f.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            f.save(file_path.replace("\\","/"))
        prediction = predict_tumors(file_path)
        
        # replace sumn
        start = ".."
        file_path = start + file_path[3:]
        file_path = file_path.replace("\\","/")
        #return file_path
        return render_template('tumors-result.html', prediction=prediction, file_path=file_path)
    return " "
