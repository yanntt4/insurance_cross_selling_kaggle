# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 10:00:27 2023

@author: ythiriet
"""

# Global librairies
from flask import Flask, request, render_template, send_file, redirect, url_for
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash
import random
import os
import sys
import numpy as np
import pandas as pd
import shutil
import openpyxl
import os
from zipfile import ZipFile

# Local
CURRENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{CURRENT_DIRECTORY}/script/")
import displaying

# App object creation
app = Flask(__name__,
            static_url_path='/static')

# Security object creation
auth = HTTPBasicAuth()

# Password authorized
user = "admin"
pw = "admin"
users = {
    user: generate_password_hash(pw)
}

# Decorator with the verify password function
@auth.verify_password
def verify_password(username, password):
    if username in users:
        return check_password_hash(users.get(username), password)
    return False


# Drop Off page
@app.route("/insurance_cross_selling")
@auth.login_required
def predict():
    displaying.preparation()
    return render_template("predict.html")


# Page Getting Files Uploaded
@app.route("/treatment", methods = ["GET", "POST"])
def treatment():

    # Init
    DATA_NAMES = ["Gender", "Age", "Driving_License", "Region_Code",
                  "Previously_Insured","Vehicle_Age","Vehicle_Damage",
                  "Annual_Premium","Vintage"]
    MODEL_INPUT = np.zeros([len(DATA_NAMES)], dtype = object)

    if request.method == "POST":

        for i, NAME in enumerate(DATA_NAMES):
            # Getting infos from predict page
            try:
                MODEL_INPUT[i] = request.form[NAME]
            except Exception as e:
                print(e)
                return render_template("Erreur.html")

        # Making prediction
        displaying.prediction(CURRENT_DIRECTORY, MODEL_INPUT, DATA_NAMES)
        return render_template("result.html")

    # Issue when collecting data
    return render_template("Erreur.html")

# Launching the Server
if __name__ == "__main__":
    app.run(debug=True)
