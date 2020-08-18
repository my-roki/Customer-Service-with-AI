import requests
from bs4 import BeautifulSoup
from flask import Flask, render_template, request
from DB_Controller import timely_customer_count, today_count, week_count, month_count, menu_count
from DB_Service import user_history

app = Flask(__name__)
 
@app.route('/')
def index():
  return render_template('login.html')
  
@app.route('/home')
def home():
  time_count = timely_customer_count()
  customer_count= [today_count(), week_count(), month_count()]
  m_count = menu_count()
  return render_template('home.html', counts = time_count, cust_count = customer_count, menu_count=m_count) 

@app.route('/service')
def service():
  user_no = 1
  u_history = user_history(user_no)
  return render_template('service.html', u_data = u_history) 
  
@app.route('/contact')
def contact():
  return render_template('contact.html') 

@app.route('/about')
def about():
  return render_template('about.html')
  
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=80)
