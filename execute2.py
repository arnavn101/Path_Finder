while(True):
    import os
    import time
    import sys
    exists = os.path.isfile('execute2.py')
    if exists:
        from ann2_predict import ann_predict
        
        
        
        from flask import Flask
        from flask import request
        from flask import render_template
        
        from flask import Flask
        
        def arp():
            return ann_predict('Jan', 'CS', 1400, 111, 10, 19, 101,1,1,1,'yes',90)
        
        
        hi = arp()
        
        
        app = Flask(__name__)
        
        
        @app.route('/')
        def hello_world():
            return hi
        

        if __name__ == '__main__':
            app.run()          
    else:
        time.sleep(1)
