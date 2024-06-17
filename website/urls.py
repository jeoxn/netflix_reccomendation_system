from flask import Blueprint, render_template, request
from .main import reccomend

urls = Blueprint('urls', __name__)

@urls.route('/', methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        try:
            title = request.form['title']
            recommendations = reccomend(title)
            return render_template('index.html', recommendations=recommendations, title=title)
        except:
            return render_template('index.html', error=True)
    return render_template('index.html')