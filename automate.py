from flask import Flask, render_template, request
from collegerecommender import College_Recommender
from AcceptancePredictor import Acceptance_Predictor
import os

app = Flask(__name__)

@app.route('/result', methods = ['POST', 'GET'])
def result():
    if request.method == 'POST':
        identifier = request.form.get("formIdentify")
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

        if identifier == "acceptance":
            college = request.form.get("College")
            resulto = Acceptance_Predictor("data_set.csv", doa, field, sat, gre, awa, toefl,ielts,experience,loan,papers,inter,grade, college)
            response = resulto.return_respose()
            return render_template('results.html', value=response)
        elif identifier == "recommender":
            resulto =  College_Recommender("data_set.csv", doa, field, sat, gre, awa, toefl,ielts,experience,loan,papers,inter,grade) 
            response = resulto.return_respose()
            return render_template('results.html', value=response)

if __name__ == '__main__':
       app.run(host='0.0.0.0', port=5000)

    
