# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 13:12:32 2023

@author: ythiriet
"""


# Function to create HTML page to ask user data to make prediction
def preparation():


    # Global importation
    from yattag import Doc
    from yattag import indent
    import joblib
    import numpy as np

    # Data importation
    ARRAY_DATA_ENCODE_REPLACEMENT = joblib.load("./script/data_replacement/array_data_encode_replacement.joblib")
    NAME_DATA_ENCODE_REPLACEMENT = np.zeros([ARRAY_DATA_ENCODE_REPLACEMENT.shape[0]], dtype = object)
    for i, ARRAY in enumerate(ARRAY_DATA_ENCODE_REPLACEMENT):
        NAME_DATA_ENCODE_REPLACEMENT[i] = ARRAY[0,0]

    # Setting list for prediction
    GENDERS = ARRAY_DATA_ENCODE_REPLACEMENT[np.where(NAME_DATA_ENCODE_REPLACEMENT == "Gender")[0][0]][:,-1]
    VEHICLE_AGE = ARRAY_DATA_ENCODE_REPLACEMENT[np.where(NAME_DATA_ENCODE_REPLACEMENT == "Vehicle_Age")[0][0]][:,-1]

    


    # Creating HTML
    doc, tag, text, line = Doc().ttl()

    # Adding pre-head
    doc.asis('<!DOCTYPE html>')
    doc.asis('<html lang="fr">')
    with tag('head'):
        doc.asis('<meta charset="UTF-8">')
        doc.asis('<meta http-equiv="X-UA-Compatible" content = "IE=edge">')
        doc.asis('<link rel="stylesheet" href="./static/style.css">')
        doc.asis('<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-QWTKZyjpPEjISv5WaRU9OFeRpok6YctnYmDr5pNlyT2bRjXh0JMhjY6hW+ALEwIH" crossorigin="anonymous">')
        doc.asis('<meta name = "viewport" content="width=device-width, initial-scale = 1.0">')

    # Body start
    with tag('body', klass = 'background_green_prolinair_light'):
        with tag('div', klass = "container"):
            with tag('div', klass = "row"):
                with tag('div', klass = "col-md-9 frame_content"):
                    line('h1', 'Interet du client pour une assurance voiture en plus de son assurance sante ?', klass = "text-center title blue_prolinair")
                with tag('div', klass = "col"):
                    doc.asis('<img src="/static/insurance.jpg" alt="Insurance" width=100% height=100% title="Insurance"/>')
                
            # Launching prediction
            with tag('form', action = "{{url_for('treatment')}}", method = "POST", enctype = "multipart/form-data"):
                
                # Gender
                with tag('div', klass = "row frame_content"): 
                    with tag('div', klass = "col"):
                        with tag('div', klass = "row"): 
                            with tag('div', klass = "col-md-6"):
                                with tag('div', klass = "row justify-content-between"): 
                                    with tag('div', klass = "col-md-3"):
                                        line('p', 'Genre', klass = "list_choice purple font_family_monospace")
                                    with tag('div', klass = "col", style="text-align:center"):
                                        with tag('div', klass = "row"):
                                            for GENDER in GENDERS:
                                                with tag('div', klass = "col-md-4"):
                                                    doc.input(name = 'Gender', type = 'radio', value = GENDER, klass = 'radio_text')
                                                    text(GENDER)
        
                        # Age
                        line('hr','')
                        with tag('div', klass = "row"): 
                            with tag('div', klass = "col-md-6"):
                                with tag('div', klass = "row justify-content-between"):
                                    with tag('div', klass = "col-md-6"):
                                        line('p', 'Age', klass = "list_choice purple font_family_monospace")
                                    with tag('div', klass = "col", style="text-align:center"):
                                        doc.input(name = 'Age', type = 'text', size = "4", placeholder = "0",
                                                  minlength = 1, klass = 'area_input')
        
                            # Driving Licence
                            with tag('div', klass = "col-md-6"):
                                with tag('div', klass = "row justify-content-between"): 
                                    with tag('div', klass = "col-md-6"):
                                        line('p', 'Permis de conduire', klass = "list_choice purple font_family_monospace")
                                    with tag('div', klass = "col", style="text-align:center"):
                                        with tag('div', klass = "row"):
                                            with tag('div', klass = "col-md-4"):
                                                doc.input(name = 'Driving_License', type = 'radio', value = 0, klass = 'radio_text')
                                                text("NON")
                                            with tag('div', klass = "col-md-4"):
                                                doc.input(name = 'Driving_License', type = 'radio', value = 1, klass = 'radio_text')
                                                text("OUI")
                        
                        # Region code
                        line('hr','')
                        with tag('div', klass = "row"): 
                            with tag('div', klass = "col-md-6"):
                                with tag('div', klass = "row justify-content-between"):
                                    with tag('div', klass = "col-md-6"):
                                        line('p', 'Etat Americain habite', klass = "list_choice purple font_family_monospace")
                                    with tag('div', klass = "col", style="text-align:center"):
                                        doc.input(name = 'Region_Code', type = 'text', size = "4", placeholder = "0",
                                                  minlength = 1, klass = 'area_input')
                        
                        # Previously Insured
                            with tag('div', klass = "col-md-6"):
                                with tag('div', klass = "row justify-content-between"): 
                                    with tag('div', klass = "col-md-6"):
                                        line('p', 'Deja eu une assurance voiture', klass = "list_choice purple font_family_monospace")
                                    with tag('div', klass = "col", style="text-align:center"):
                                        with tag('div', klass = "row"):
                                            with tag('div', klass = "col-md-4"):
                                                doc.input(name = 'Previously_Insured', type = 'radio', value = 0, klass = 'radio_text')
                                                text("NON")
                                            with tag('div', klass = "col-md-4"):
                                                doc.input(name = 'Previously_Insured', type = 'radio', value = 1, klass = 'radio_text')
                                                text("OUI")
                        
                        # Vehicle Age
                        line('hr','')
                        with tag('div', klass = "row justify-content-between"): 
                            with tag('div', klass = "col-md-6"):
                                line('p', 'Age du vehicule', klass = "list_choice purple font_family_monospace")
                            with tag('div', klass = "col", style="text-align:center"):
                                with tag('div', klass = "row"):
                                    for AGE in VEHICLE_AGE:
                                        with tag('div', klass = "col-md-4"):
                                            doc.input(name = 'Vehicle_Age', type = 'radio', value = AGE, klass = 'radio_text')
                                            text(AGE)
                        
                        # Previously Damaged
                        line('hr','')
                        with tag('div', klass = "row justify-content-between"): 
                            with tag('div', klass = "col-md-6"):
                                line('p', 'Deja eu un accident de voiture', klass = "list_choice purple font_family_monospace")
                            with tag('div', klass = "col", style="text-align:center"):
                                with tag('div', klass = "row"):
                                    for RESULT in ["OUI","NON"]:
                                        with tag('div', klass = "col-md-4"):
                                            doc.input(name = 'Vehicle_Damage', type = 'radio', value = RESULT, klass = 'radio_text')
                                            text(RESULT)
                        
                        # Annual Premium
                        line('hr','')
                        with tag('div', klass = "row"): 
                            with tag('div', klass = "col-md-6"):
                                with tag('div', klass = "row justify-content-around margin_1"): 
                                    with tag('div', klass = "col-md-6"):
                                        line('p', "Cout de l'assurance sante", klass = "list_choice purple font_family_monospace")
                                    with tag('div', klass = "col", style="text-align:center"):
                                        doc.input(name = 'Annual_Premium', type = 'text', size = "4", placeholder = "0",
                                                  minlength = 1, klass = 'area_input')
                        
                        # Vintage (current contract duration)
                            with tag('div', klass = "col-md-6"):
                                with tag('div', klass = "row justify-content-around margin_1"): 
                                    with tag('div', klass = "col-md-8"):
                                        line('p', "Duree depuis la 1ere souscription [en jours]", klass = "list_choice purple font_family_monospace")
                                    with tag('div', klass = "col", style="text-align:center"):
                                        doc.input(name = 'Vintage', type = 'text', size = "4", placeholder = "0",
                                                  minlength = 1, klass = 'area_input')
        
                        # Submit button
                        with tag('div', klass = "row"):
                            with tag('div', klass = "text-center div2"):
                                with tag('button', id = 'submit_button', name = "action", klass="btn btn-primary", value = 'Predict'):
                                    text('Realiser une prediction')
                                

    # Saving HTML created
    with open(f"./templates/predict.html", "w") as f:
        f.write(indent(doc.getvalue(), indentation = '    ', newline = '\n', indent_text = True))
        f.close()


# Function to make prediction and plotting them for the customer
def prediction(CURRENT_DIRECTORY, MODEL_INPUT_HTML, DATA_NAMES_HTML):

    # Global importation
    import joblib
    import numpy as np
    from yattag import Doc
    from yattag import indent
    
    # Global init
    RF_MODEL = False
    NN_MODEL = False
    GB_MODEL = False
    XG_MODEL = True
    REGRESSION = False

    # Class creation
    class Data_prediction():
        def __init__(self, MODEL):
            self.ARRAY_DATA_ENCODE_REPLACEMENT = joblib.load("./script/data_replacement/array_data_encode_replacement.joblib")
            self.DATA_NAMES = joblib.load("./script/data_replacement/data_names.joblib")
            
            self.MODEL = MODEL
            
            self.JS_CANVAS = ""
            self.JS_ANIMATION = ""

        
        def entry_data_arrangement(self, MODEL_INPUT_HTML, DATA_NAMES_HTML):
            self.MODEL_INPUT = np.zeros([self.DATA_NAMES.shape[0]], dtype = object)
            DATA_NAMES_HTML = np.array(DATA_NAMES_HTML)
            
            for i, DATA_NAME in enumerate(self.DATA_NAMES):
                print(DATA_NAME)
                print(DATA_NAMES_HTML)
                self.MODEL_INPUT[i] = MODEL_INPUT_HTML[np.where(DATA_NAME == DATA_NAMES_HTML)[0][0]]


        # Turning word into numbers to make predictions
        def entry_data_modification(self):

            for i in range(self.MODEL_INPUT.shape[0]):
                for ARRAY in self.ARRAY_DATA_ENCODE_REPLACEMENT:
                    if self.DATA_NAMES[i] == ARRAY[0,0]:
                        for j in range(ARRAY.shape[0]):
                            if self.MODEL_INPUT[i] == ARRAY[j,3]:
                                self.MODEL_INPUT[i] = ARRAY[j,2]


        # Making prediction using model chosen
        def making_prediction(self, REGRESSION):
            print(self.MODEL_INPUT)
            self.PREDICTION = self.MODEL.predict(self.MODEL_INPUT.reshape(1,-1))
            print(self.PREDICTION)
            
            if REGRESSION == False:
                self.PROBA = self.MODEL.predict_proba(self.MODEL_INPUT.reshape(1,-1))
                print(self.PROBA)


        # Creating javascript using prediction
        def javascript_result_creation(self):
            
            # Creating Canvas to plot graphic
            self.JS_CANVAS += '/* Creation du canvas */\n'
            self.JS_CANVAS += 'var canvas = document.getElementById("canvas1");\n'
            self.JS_CANVAS += 'const width = (canvas.width = window.innerWidth);\n'
            self.JS_CANVAS += 'const height = (canvas.height = 400);\n'
            self.JS_CANVAS += f'const x = {int(self.PREDICTION[0]/1000)};\n'
            self.JS_CANVAS += "canvas.style.position = 'relative';\n"
            self.JS_CANVAS += "canvas.style.zIndex = 1;"
            for i in range(12):
                self.JS_CANVAS += f'var ctx{i} = canvas.getContext("2d");\n'
            
            # Function to display correct format for number
            self.JS_CANVAS += '\n/* Fonction pour modifier le style des nombres affichés */\n'
            self.JS_CANVAS += 'function numberWithCommas(x) {\n'
            self.JS_CANVAS += '   return x.toString().replace(/\B(?=(\d{3})+(?!\d))/g, " ");\n'
            self.JS_CANVAS += '}\n'
            
            # Function to display correct format for number
            self.JS_CANVAS += '\n/* Fonction pour generer un entier aleatoire entre 2 valeurs */\n'
            self.JS_CANVAS += 'function getRandomInt(min, max) {\n'
            self.JS_CANVAS += '    const minCeiled = Math.ceil(min);\n'
            self.JS_CANVAS += '    const maxFloored = Math.floor(max);\n'
            self.JS_CANVAS += '    return Math.floor(Math.random() * (maxFloored - minCeiled) + minCeiled);\n'
            self.JS_CANVAS += '}\n'
            
            # Function to change color
            self.JS_CANVAS += "\n/* Creation d'une fonction pour changer la couleur */\n"
            self.JS_CANVAS += 'function rgb(r, g, b){\n'
            self.JS_CANVAS += 'return "rgb("+r+","+g+","+b+")";\n'
            self.JS_CANVAS += '}\n'
            
            # Car construction functions
            self.JS_CANVAS += "\n/* Fonction pour creer une formule 1 orientee vers la gauche */\n"
            self.JS_CANVAS += 'function create_auto_drawing_left(wheel_position_x, wheel_position_y) {\n'
            self.JS_CANVAS += '    ctx2.strokeStyle = "rgb(0,0,0)";\n'
            self.JS_CANVAS += '    ctx2.beginPath();\n'
            self.JS_CANVAS += '    ctx2.arc(wheel_position_x,wheel_position_y,12,0,2*Math.PI,false);\n'
            self.JS_CANVAS += '    ctx2.stroke();\n'
            self.JS_CANVAS += '    ctx2.beginPath();\n'
            self.JS_CANVAS += '    ctx2.arc(wheel_position_x + 120,wheel_position_y,12,0,2*Math.PI,false);\n'
            self.JS_CANVAS += '    ctx2.stroke();\n'
            self.JS_CANVAS += '    ctx2.beginPath();\n'
            self.JS_CANVAS += '    ctx2.moveTo(wheel_position_x - 12,wheel_position_y + 4);\n'
            self.JS_CANVAS += '    ctx2.lineTo(wheel_position_x - 50,wheel_position_y + 4);\n'
            self.JS_CANVAS += '    ctx2.bezierCurveTo(wheel_position_x - 25,wheel_position_y - 16,wheel_position_x,wheel_position_y - 19,wheel_position_x + 25,wheel_position_y - 21);\n'
            self.JS_CANVAS += '    ctx2.bezierCurveTo(wheel_position_x + 55,wheel_position_y - 20,wheel_position_x + 65,wheel_position_y - 18,wheel_position_x + 75,wheel_position_y - 16);\n'
            self.JS_CANVAS += '    ctx2.bezierCurveTo(wheel_position_x + 85,wheel_position_y - 14,wheel_position_x + 95,wheel_position_y - 12,wheel_position_x + 105,wheel_position_y - 11);\n'
            self.JS_CANVAS += '    ctx2.stroke();\n'
            self.JS_CANVAS += '    ctx2.beginPath();\n'
            self.JS_CANVAS += '    ctx2.moveTo(wheel_position_x + 13,wheel_position_y + 4);\n'
            self.JS_CANVAS += '    ctx2.lineTo(wheel_position_x + 102,wheel_position_y + 4);\n'
            self.JS_CANVAS += '    ctx2.stroke();\n'
            self.JS_CANVAS += '    ctx2.beginPath();\n'
            self.JS_CANVAS += '    ctx2.moveTo(wheel_position_x + 55,wheel_position_y - 20);\n'
            self.JS_CANVAS += '    ctx2.lineTo(wheel_position_x + 55,wheel_position_y - 35);\n'
            self.JS_CANVAS += '    ctx2.bezierCurveTo(wheel_position_x + 59,wheel_position_y - 37,wheel_position_x + 64,wheel_position_y - 38.3,wheel_position_x + 67,wheel_position_y - 38);\n'
            self.JS_CANVAS += '    ctx2.bezierCurveTo(wheel_position_x + 70,wheel_position_y - 38,wheel_position_x + 80,wheel_position_y - 36,wheel_position_x + 90,wheel_position_y - 34);\n'
            self.JS_CANVAS += '    ctx2.bezierCurveTo(wheel_position_x + 100,wheel_position_y - 31,wheel_position_x + 120,wheel_position_y - 29,wheel_position_x + 140,wheel_position_y - 28);\n'
            self.JS_CANVAS += '    ctx2.stroke();\n'
            self.JS_CANVAS += '    ctx2.beginPath();\n'
            self.JS_CANVAS += '    ctx2.moveTo(wheel_position_x + 130,wheel_position_y - 6);\n'
            self.JS_CANVAS += '    ctx2.bezierCurveTo(wheel_position_x + 145,wheel_position_y - 11,wheel_position_x + 147,wheel_position_y - 22,wheel_position_x + 150,wheel_position_y - 37);\n'
            self.JS_CANVAS += '    ctx2.lineTo(wheel_position_x + 130,wheel_position_y - 37);\n'
            self.JS_CANVAS += '    ctx2.stroke();\n'
            self.JS_CANVAS += '    ctx2.lineWidth = 1;\n'
            self.JS_CANVAS += '}\n'
            
            self.JS_CANVAS += "\n/* Fonction pour creer une formule 1 orientee vers la droite */\n"
            self.JS_CANVAS += 'function create_auto_drawing_right(wheel_position_x, wheel_position_y) {\n'
            self.JS_CANVAS += '    ctx2.strokeStyle = "rgb(0,0,0)";\n'
            self.JS_CANVAS += '    ctx2.beginPath();\n'
            self.JS_CANVAS += '    ctx2.arc(wheel_position_x,wheel_position_y,12,0,2*Math.PI,false);\n'
            self.JS_CANVAS += '    ctx2.stroke();\n'
            self.JS_CANVAS += '    ctx2.beginPath();\n'
            self.JS_CANVAS += '    ctx2.arc(wheel_position_x - 120,wheel_position_y,12,0,2*Math.PI,false);\n'
            self.JS_CANVAS += '    ctx2.stroke();\n'
            self.JS_CANVAS += '    ctx2.beginPath();\n'
            self.JS_CANVAS += '    ctx2.moveTo(wheel_position_x + 12,wheel_position_y + 4);\n'
            self.JS_CANVAS += '    ctx2.lineTo(wheel_position_x + 50,wheel_position_y + 4);\n'
            self.JS_CANVAS += '    ctx2.bezierCurveTo(wheel_position_x + 25,wheel_position_y - 16,wheel_position_x,wheel_position_y - 19,wheel_position_x - 25,wheel_position_y - 21);\n'
            self.JS_CANVAS += '    ctx2.bezierCurveTo(wheel_position_x - 55,wheel_position_y - 20,wheel_position_x - 65,wheel_position_y - 18,wheel_position_x - 75,wheel_position_y - 16);\n'
            self.JS_CANVAS += '    ctx2.bezierCurveTo(wheel_position_x - 85,wheel_position_y - 14,wheel_position_x - 95,wheel_position_y - 12,wheel_position_x - 105,wheel_position_y - 11);\n'
            self.JS_CANVAS += '    ctx2.stroke();\n'
            self.JS_CANVAS += '    ctx2.beginPath();\n'
            self.JS_CANVAS += '    ctx2.moveTo(wheel_position_x - 13,wheel_position_y + 4);\n'
            self.JS_CANVAS += '    ctx2.lineTo(wheel_position_x - 102,wheel_position_y + 4);\n'
            self.JS_CANVAS += '    ctx2.stroke();\n'
            self.JS_CANVAS += '    ctx2.beginPath();\n'
            self.JS_CANVAS += '    ctx2.moveTo(wheel_position_x - 55,wheel_position_y - 20);\n'
            self.JS_CANVAS += '    ctx2.lineTo(wheel_position_x - 55,wheel_position_y - 35);\n'
            self.JS_CANVAS += '    ctx2.bezierCurveTo(wheel_position_x - 59,wheel_position_y - 37,wheel_position_x - 64,wheel_position_y - 38.3,wheel_position_x - 67,wheel_position_y - 38);\n'
            self.JS_CANVAS += '    ctx2.bezierCurveTo(wheel_position_x - 70,wheel_position_y - 38,wheel_position_x - 80,wheel_position_y - 36,wheel_position_x - 90,wheel_position_y - 34);\n'
            self.JS_CANVAS += '    ctx2.bezierCurveTo(wheel_position_x - 100,wheel_position_y - 31,wheel_position_x - 120,wheel_position_y - 29,wheel_position_x - 140,wheel_position_y - 28);\n'
            self.JS_CANVAS += '    ctx2.stroke();\n'
            self.JS_CANVAS += '    ctx2.beginPath();\n'
            self.JS_CANVAS += '    ctx2.moveTo(wheel_position_x - 130,wheel_position_y - 6);\n'
            self.JS_CANVAS += '    ctx2.bezierCurveTo(wheel_position_x - 145,wheel_position_y - 11,wheel_position_x - 147,wheel_position_y - 22,wheel_position_x - 150,wheel_position_y - 37);\n'
            self.JS_CANVAS += '    ctx2.lineTo(wheel_position_x - 130,wheel_position_y - 37);\n'
            self.JS_CANVAS += '    ctx2.stroke();\n'
            self.JS_CANVAS += '    ctx2.lineWidth = 1;\n'
            self.JS_CANVAS += '}\n'
            
            # Car animation
            self.JS_CANVAS += "\n/* Creation d'une animation pour faire bouger la voiture */\n"
            self.JS_CANVAS += 'var id = null;\n'
            self.JS_CANVAS += 'function myMove() {\n'
            self.JS_CANVAS += '  var pos = 150;\n'
            self.JS_CANVAS += f'  var proba = {self.PROBA[0][1]};\n'
            self.JS_CANVAS += f'  var proba_percent = {int(100*self.PROBA[0][1])};\n'
            self.JS_CANVAS += '   const x = pos + 1000*proba;\n'
            self.JS_CANVAS += '  clearInterval(id);\n'
            self.JS_CANVAS += '  id = setInterval(frame, 10);\n'
            self.JS_CANVAS += '  function frame() {\n'
            self.JS_CANVAS += '    if (pos > x) {\n'
            self.JS_CANVAS += '      ctx2.clearRect(0,0,width,190);\n'
            self.JS_CANVAS += '      create_auto_drawing_right(pos,150);\n'
            self.JS_CANVAS += '      ctx2.font = "38px georgia";\n'
            self.JS_CANVAS += '     ctx2.strokeText(`${proba_percent} %`,pos - 100,90);\n'
            self.JS_CANVAS += '    } else {\n'
            self.JS_CANVAS += '      ctx2.clearRect(0,0,width,190);\n'
            self.JS_CANVAS += '      pos += getRandomInt(1,5);\n'
            self.JS_CANVAS += '      create_auto_drawing_right(pos,150);\n'
            self.JS_CANVAS += '    }\n'
            self.JS_CANVAS += '  }\n'
            self.JS_CANVAS += '}\n'
            
            # Percentage Line
            self.JS_CANVAS += 'ctx3.strokeStyle = "rgb(0,0,0)";\n'
            self.JS_CANVAS += 'var line_height = 200;\n'
            self.JS_CANVAS += 'ctx3.lineWidth = 1;\n'
            self.JS_CANVAS += 'ctx4.lineWidth = 3;\n'
            self.JS_CANVAS += 'ctx5.font = "28px georgia";\n'
            self.JS_CANVAS += 'ctx3.beginPath();\n'
            self.JS_CANVAS += 'ctx3.moveTo(100,line_height);\n'
            self.JS_CANVAS += 'ctx3.lineTo(1200,line_height);\n'
            self.JS_CANVAS += 'ctx3.lineTo(1190,line_height - 10);\n'
            self.JS_CANVAS += 'ctx3.lineTo(1200,line_height);\n'
            self.JS_CANVAS += 'ctx3.lineTo(1190,line_height + 10);\n'
            self.JS_CANVAS += 'ctx5.strokeText("[%]",1200,line_height + 33);\n'
            self.JS_CANVAS += 'ctx5.strokeText("Pourcentage de chance que le client soit interesse",width/6,line_height + 100);\n'
            self.JS_CANVAS += 'ctx3.stroke();\n'
            for i in [200,300,400,500,600,700,800,900,1000,1100]:
                self.JS_CANVAS += 'ctx4.beginPath();\n'
                self.JS_CANVAS += f'ctx4.moveTo({i},line_height + 10);\n'
                self.JS_CANVAS += f'ctx4.lineTo({i},line_height - 10);\n'
                
                self.JS_CANVAS += f'ctx5.strokeText({(i-100)/10},{i - 15},line_height + 33);\n'
                self.JS_CANVAS += 'ctx4.stroke();\n'
    
            
            # Writing Javascript into a file
            with open("./static/canevas.js","w") as f:
                f.write(self.JS_CANVAS)
    

        # Creating html
        def html_result_creation(self, CURRENT_DIRECTORY):

            # Creating HTML
            doc, tag, text, line = Doc(defaults = {'Month': 'Fevrier'}).ttl()

            doc.asis('<!DOCTYPE html>')
            doc.asis('<html lang="fr">')
            with tag('head'):
                doc.asis('<meta charset="UTF-8">')
                doc.asis('<meta http-equiv="X-UA-Compatible" content = "IE=edge">')
                doc.asis('<link rel="stylesheet" href="./static/style.css">')
                doc.asis('<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-QWTKZyjpPEjISv5WaRU9OFeRpok6YctnYmDr5pNlyT2bRjXh0JMhjY6hW+ALEwIH" crossorigin="anonymous">')
                doc.asis('<meta name = "viewport" content="width=device-width, initial-scale = 1.0">')

            # Body start
            with tag('body', klass = 'background_green_prolinair_light'):
                with tag('div', klass = "container"):
                    with tag('div', klass = "row"):
                        with tag('div', klass = "col-md-9 frame_content"):
                            line('h1', 'Interet du client pour une assurance voiture en plus de son assurance sante ?', klass = "text-center title blue_prolinair")
                        with tag('div', klass = "col"):
                            doc.asis('<img src="/static/insurance.jpg" alt="Insurance" width=100% height=100% title="Insurance"/>')


                    with tag('div', klass = "row"):
                        with tag('div', klass="col frame_content"):
                            with tag('canvas', id = "canvas1", width="540", height="400"):
                                text("")
                        
                                # Script for canvas
                                doc.asis('<script src="/static/canevas.js"></script>')
                    
                            # Launching script when arriving on the page
                            with tag('script', type="text/javascript"):
                                text('myMove();')
                
                # Button to go back to previous page
                    with tag('form', action = "{{url_for('predict')}}", method = "GET", enctype = "multipart/form-data"):
                        with tag('div', klass = "text-center"):
                            with tag('button', id = 'submit_button', name = "action", klass="btn btn-primary", value = 'Go back to previous page'):
                                line('p1', '')
                                text('Retourner a la page precedente')

            # Saving HTML
            with open("./templates/result.html", "w") as f:
                f.write(indent(doc.getvalue(), indentation = '    ', newline = '\n', indent_text = True))
                f.close()

    # Loading models
    if RF_MODEL == True:
        with open("./script/models/rf_model.sav", 'rb') as f:
            MODEL = joblib.load(f)
    elif NN_MODEL == True:
        with open("./script/models/nn_model.sav", 'rb') as f:
            MODEL = joblib.load(f)
    elif GB_MODEL == True:
        with open("./script/models/gb_model.sav", 'rb') as f:
            MODEL = joblib.load(f)
    elif XG_MODEL == True:
        with open("./script/models/xg_model.sav", 'rb') as f:
            MODEL = joblib.load(f)

    # Personnalized prediction
    global_data_prediction = Data_prediction(MODEL)
    global_data_prediction.entry_data_arrangement(MODEL_INPUT_HTML, DATA_NAMES_HTML)
    global_data_prediction.entry_data_modification()

    global_data_prediction.making_prediction(REGRESSION)
    global_data_prediction.javascript_result_creation()
    global_data_prediction.html_result_creation(CURRENT_DIRECTORY)
