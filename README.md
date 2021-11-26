# Path_Finder

********
***Overview***
********

College counseling takes up so much money, approx. 5000 to 20000 dollars a year. All this money goes to a person who determines the path best fit for you, however, many miss out on professional advice like this and miss their best path due to the expense. So we thought, why not make counseling affordable? In fact, why not make it completely free of charge?


- ``Path_Finder``: Shows an optimized path to college
 

Functionality
=====

Fundamentally, ``Path_Finder`` is capable of 
    
    1) Using neural networks and machine learning to train on admissions data on students
  
    2) Show a student's probability for a college based their grades and extracurriculars
    
    3) Recommend the college that a student is most likely best fit for based on their profile
    
    4) Smartly recognizing changes in the college admissions process using data instead of pre-defined metrics


************
Installation/Configuration
************

Linux Environments
==========================

On any Linux OS, clone this repository 

     git clone https://github.com/arnavn101/NotesCreator.git
      

Install requirements of Python
    
    pip install keras Flask matplotlib pandas sklearn numpy
         

Run the Main File: 
     
     python3 automate.py

Testing Output with CURL
```bash 
curl -X POST  "http://127.0.0.1:5000/" \
              -d identifier=acceptance -d College=MIT \
              -d sat=1600 \
               -d grade=100 > index.html

firefox index.html # Can be any other browser
```


APIs and Resources Used
===============
      keras
      sklearn
      Flask
      pandas
      numpy
      # For more, refer to the collegerecommender.py


