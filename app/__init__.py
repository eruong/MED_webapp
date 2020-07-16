# Authors: Melody, Ethan, Dylan
try:
    from flask import Flask
    import gunicorn
except:
    print('Make sure to pip install Flask twilio')
from config import Config

app = Flask(__name__, static_url_path='')
app.config.from_object(Config)

from app import routes