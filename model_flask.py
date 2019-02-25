from flask import Flask,request,jsonify
from sklearn.externals import joblib
import traceback
import pandas as pd
import numpy as np
import sys

# API definition

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def p():
    if rf:
        try:
            json_ = request.json
            print(json_)
            print("after json request")
            query = pd.get_dummies(pd.DataFrame(json_))
            query = query.reindex(columns = model_columns, fill_value = 0)

            prediction = list(rf.predict(query))

            return jsonify({"prediction": str(prediction)})

        except:
            return jsonify({"trace": traceback.format_exc()})


    else:
         print("Train the model first")
         return ("no model here to use")




if(__name__ == "__main__"):
    
#     try:
#          port = int(sys.argv[1]) # This is for a command-line input
#     except:
#         port = 12345 # If you don't provide any port the port will be set to 12345

    rf = joblib.load("model.pkl") # Load "model.pkl"
    print ('Models loaded')
    print(rf)
    
    model_columns = joblib.load("model_columns.pkl") # Load "model_columns.pkl"
    print ('Models columns loaded')
    print(model_columns)

    app.run(host='0.0.0.0', port=80)


    
