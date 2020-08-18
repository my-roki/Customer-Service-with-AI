import requests
from bs4 import BeautifulSoup
from flask import Flask, render_template, request

app = Flask(__name__)
 
@app.route('/')
def index():
  return render_template('login.html')
  
@app.route('/home')
def home():
  return render_template('home.html') 

@app.route('/service')
def service():
  return render_template('service.html') 
  
@app.route('/contact')
def contact():
  return render_template('contact.html') 

@app.route('/about')
def about():
  return render_template('about.html')
  
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=80)
