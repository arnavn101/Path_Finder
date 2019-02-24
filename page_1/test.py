from flask import Flask, render_template, request
from google_images_download import google_images_download
response = google_images_download.googleimagesdownload()
app = Flask(__name__)

@app.route('/')
def data():
   return render_template('test.html')

@app.route('/result', methods = ['POST', 'GET'])
def result():
   if request.method == 'POST':
      
      doa = request.form.get("Date_of_Application")
      field = request.form.get("Field")
      sat = request.form.get("SAT_score")
      gre = request.form.get("GRE_score")
      awa = request.form.get("AWA_score")
      toefl = request.form.get("TOEFL_score")
      ielts = request.form.get("IELTS_score")
      experience = request.form.get("Work_experience")
      papers = request.form.get("International_papers")
      loan = request.form.get("undergraduate_loan")
      inter = request.form.get("int_or_not")
      grade = request.form.get("id")
      
      
      from ann2_predict import ann2_predict
      code =  ann2_predict(doa, field, sat, gre, awa, toefl,ielts,experience,loan,papers,inter,grade) 
      return code + render_template('results.html')
if __name__ == '__main__':
   app.run(host='127.0.0.1',port=12345)
