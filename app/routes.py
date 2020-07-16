# Authors: Melody, Ethan, Dylan
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
        if result == "Malott":
            description = "Located at the corner of Kravis and Scripps, students really like their salmon and steak Thursday nights. Also they bake fresh chocolate cookies every weekday at 6pm!"
            d_hall = "https://lh3.googleusercontent.com/proxy/Ja-imszkckbhPZzMHh0GtMPHaAKqdDKtSIYi8SLpsLdosBCn1HdNEsoan8MeosKt3lsAB4mdffmMncpAJfC_aQqgOh8paq2so2y_xpPNL9jjFVBndijzuruFSKJabibnMnaX343G9zcJdFK4SAOhjrORxQuS6QuzeoUQ3gHGwVwjojQusFX1wuVDqxcqgPw"
        elif result == "Collins":
            description = "Recently rennovated in 2017, their new space allows for increased efficiency. Their grill has anything from chicken to pasta to Mongolian stir-fry!"
            d_hall = "https://localist-images.azureedge.net/photos/397593/card/3272d6d53dd2b3f3a138998dbf8d6aaff88ed8f1.jpg"
        elif result == "McConnell":
            description = "Known for their long lines on Pad Thai Fridays, McConell serves a variety of options. Along with Taco Tuesdays and Pasta Saturdays, it is frequented by 5Cs students all the time!"
            d_hall = "https://www.pitzer.edu/human-resources/wp-content/uploads/sites/48/2019/02/McConnellCenter_WestSide_002.jpg"
        elif result == "The Hoch":
            description = "Officially called the Hoch-Shanahan Dining Commons, most students at the 5Cs would call this dining hall their favorite. Most notable offerings include Build-Your-Own-Pizza Friday, Muddgolian Monday, steak to order alson on Monday, and Build-Your-Own-Smoothie during weekday breakfasts!"
            d_hall = "https://www.hmc.edu/about-hmc/wp-content/uploads/sites/2/2014/08/H-S-diners-web1.jpg"
        elif result == "Frank":
            description = "Most frequented by Pomona first-years because of it's five minute walking estimation from the south campus dormitories, Frank brunch is perhaps the most notable and overhyped meal at the 5Cs!"
            d_hall = "https://s3.amazonaws.com/secretsaucefiles/photos/images/000/110/567/large/FRANK.png?1485379096"
        elif result == "Frary":
            description = "The 5Cs flagship Harry Potter dining hall, local schools often visit the dining hall to observe the unique architecture. Apart from the mediocore food, Frary does boast several fries seasonings on the daily!"
            d_hall = "https://www.pomona.edu/sites/default/files/images/insets/frary-dining-hall.jpg"
        elif result == "Oldenborg":
            description = "The only language dining hall at the 5Cs and only open at lunch. While the food may provide an inconsistent experience, you can be sure to bond with community members while practicing your foreign language of choice!"
            d_hall = "https://www.pomona.edu/sites/default/files/styles/in_content_slide/public/oldenborg-center.jpg?itok=itqk2Cnb"
        else:
            description = "We couldn't make a decision, <a href='/quiz'>try again</a>."
            d_hall = "https://img.pngio.com/patrick-star-sad-face-gifs-tenor-sad-patrick-star-220_181.gif"

        return render_template('quizResults.html', result=result, description=description, d_hall=d_hall)
    return render_template('quiz.html', title='Home')

