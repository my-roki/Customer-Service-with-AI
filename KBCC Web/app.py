import requests
from bs4 import BeautifulSoup
from flask import Flask, render_template, request
from DB_Controller import timely_customer_count, today_count, week_count, month_count, menu_count
from DB_Service import user_history
from camera import VideoCamera


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
  

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
  
@app.route('/contact')
def contact():
  return render_template('contact.html') 

@app.route('/about')
def about():
  return render_template('about.html')
  
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=80)
