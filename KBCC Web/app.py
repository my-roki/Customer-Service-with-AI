import requests, os, glob, re
from bs4 import BeautifulSoup
from flask import Flask, render_template, request, Response
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
    custList = glob.glob("./static/images/today/*") # 오늘 방문한 고객의 사진이 찍힌 시간 리스트
    latest_file = max(custList, key=os.path.getctime) # 가장 마지막에 방문한 고객 찾기
    last_cust = os.path.basename(latest_file) # 마지막 방문 고객의 폴더이름만 선택
    last_cust = os.listdir(latest_file)
    user_no = last_cust[0][:-4]
    u_history = user_history(user_no)
    if os.path.isfile("."+latest_file+"/"+last_cust[0]):
        img_path="."+latest_file+"/"+last_cust[0]
    else:
        img_path="./static/images/not.jpg"
    return render_template('service.html', u_data = u_history, image_path="."+latest_file+"/"+last_cust[0]) 
  
  
@app.route('/contact')
def contact():
  return render_template('contact.html') 

@app.route('/about')
def about():
  return render_template('about.html')
  
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=80)
