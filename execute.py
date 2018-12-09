while(True):
    import os
    import time
    import sys
    exists = os.path.isfile('C:\\Users')
    if exists:
        from ann_predict import ann_predict
        
        from flask import Flask
        from flask import request
        from flask import render_template
        
        from flask import Flask
        
        def arp():
            return ann_predict('Dec', 'CS', 1600, 200, 50, 19, 101,10,100,1,'no',98, 'Carnegie Mellon University')
        
        
        
        hi = arp()
        
        
        app = Flask(__name__)
        
        
        @app.route('/')
        def hello_world():
            return hi
        
        if __name__ == '__main__':
            app.run()
        

    else:
        time.sleep(1)

    
