# Authors: CS-World Domination Summer19 - JG
try:
    from flask import render_template, redirect, url_for, request, send_from_directory, flash
except:
    print("Make sure to pip install Flask twilio")
from app import app
import os
from app import quiz_nn

# Home page, renders the index.html template
@app.route('/index')
@app.route('/')
def index():
    return render_template('index.html', title='Home')

# Quiz page
@app.route('/quiz',methods=['GET','POST'])
def quiz():
    if request.method == 'POST':
        a = int(request.form['transportation'])
        b = int(request.form['hangout'])
        c = int(request.form['sushi'])
        d = int(request.form['cookies'])
        e = int(request.form['pepo_melo'])
        f = int(request.form['village'])
        g = float(request.form['other_campus'])
        h = float(request.form['breakfast'])
        i = int(request.form['dessert'])
        j = float(request.form['distance'])
        
        result = quiz_nn.ownData(a,b,c,d,e,f,g,h,i,j)
        
        return render_template('quizResults.html',d_hall=result)
    return render_template('quiz.html', title='Home')

