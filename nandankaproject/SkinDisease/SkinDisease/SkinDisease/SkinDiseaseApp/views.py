from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
import pymysql
from django.http import HttpResponse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential, Model
from keras.models import model_from_json
import pickle
import os
import matplotlib.pyplot as plt
from django.core.files.storage import FileSystemStorage
import cv2

global cnn_algorithm, X_train, X_test, y_train, y_test
class_labels = ['Actinic Keratosis','Basal Cell Carcinoma','Dermatofibroma','Melanoma','Nevus','Pigmented Benign Keratosis',
                'Seborrheic Keratosis','Squamous Cell Carcinoma','Vascular Lesion']

def DiseasePrediction(request):
    if request.method == 'GET':
        return render(request, 'DiseasePrediction.html', {})

def DiseasePredictionAction(request):
    if request.method == 'POST' and request.FILES['t1']:
        global cnn_algorithm
        with open('model/model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            classifier = model_from_json(loaded_model_json)
        json_file.close()    
        classifier.load_weights("model/model_weights.h5")
        classifier._make_predict_function() 
        myfile = request.FILES['t1']
        fs = FileSystemStorage()
        filename = fs.save('SkinDiseaseApp/static/samples/test.png', myfile)
        image = cv2.imread('SkinDiseaseApp/static/samples/test.png')
        img = cv2.resize(image, (32,32))
        im2arr = np.array(img)
        im2arr = im2arr.reshape(1,32,32,3)
        img = np.asarray(im2arr)
        img = img.astype('float32')
        img = img/255
        preds = classifier.predict(img)
        predict = np.argmax(preds)
        img = cv2.imread('SkinDiseaseApp/static/samples/test.png')
        img = cv2.resize(img, (700,400))
        if os.path.exists('SkinDiseaseApp/static/samples/test.png'):
            os.remove('SkinDiseaseApp/static/samples/test.png')
        cv2.putText(img, 'Disease Detected & Classified as : '+class_labels[predict], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
        cv2.imshow('Disease Detected & Classified as : '+class_labels[predict], img)
        cv2.waitKey(0)
        return render(request, 'DiseasePrediction.html', {})        


def CNNtestPrediction(name, classifier, X_test, y_test):
    predict = classifier.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test = np.argmax(y_test, axis=1)
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    a = accuracy_score(y_test,predict)*100
    output = '<table border=1 align=center><tr><th><font size="" color="black">Algorithm Name</th><th><font size="" color="black">Accuracy</th>'
    output += '<th><font size="" color="black">Precision</th><th><font size="" color="black">Recall</th><th><font size="" color="black">FScore</th></tr>'
    output += '<tr><td><font size="" color="black">'+name+'</td>'
    output += '<td><font size="" color="black">'+str(a)+'</td>'
    output += '<td><font size="" color="black">'+str(p)+'</td>'
    output += '<td><font size="" color="black">'+str(r)+'</td>'
    output += '<td><font size="" color="black">'+str(f)+'</td></tr>'
    conf_matrix = confusion_matrix(y_test, predict) 
    return conf_matrix, output 


def runCNN(request):
    if request.method == 'GET':
        global cnn_algorithm, X_train, X_test, y_train, y_test
        X = np.load('model/X.txt.npy')
        Y = np.load('model/Y.txt.npy')
        print(Y)
        X = X.astype('float32')
        X = X/255
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = X[indices]
        Y = Y[indices]
        Y = to_categorical(Y)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        if os.path.exists('model/model.json'):
            with open('model/model.json', "r") as json_file:
                loaded_model_json = json_file.read()
                classifier = model_from_json(loaded_model_json)
            json_file.close()    
            classifier.load_weights("model/model_weights.h5")
            classifier._make_predict_function()       
        else:
            classifier = Sequential()
            #defining convolution neural network layer with 32 layers to filter dataset features 32 times and then max pooling will collect all important features
            classifier.add(Convolution2D(32, 3, 3, input_shape = (32, 32, 3), activation = 'relu'))
            #pooling layer to collect filtered features
            classifier.add(MaxPooling2D(pool_size = (2, 2)))
            #defining another layer with 32 filters
            classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
            classifier.add(MaxPooling2D(pool_size = (2, 2)))
            classifier.add(Flatten())
            classifier.add(Dense(output_dim = 256, activation = 'relu'))
            classifier.add(Dense(output_dim = y_train.shape[1], activation = 'softmax'))
            print(classifier.summary())
            #now compiling the model
            classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
            #now training CNN model with X and Y array data
            hist = classifier.fit(X, Y, batch_size=8, epochs=50, shuffle=True, verbose=2, validation_data=(X_test, y_test))
            classifier.save_weights('model/cnn_model_weights.h5')            
            model_json = classifier.to_json()
            with open("model/cnn_model.json", "w") as json_file:
                json_file.write(model_json)
            json_file.close()
        print(classifier.summary())    
        cnn_algorithm = classifier    
        conf_matrix, output = CNNtestPrediction("CNN Skin Disease Classification",classifier,X_test,y_test)
        context= {'data':output}
        plt.figure(figsize =(6, 6)) 
        ax = sns.heatmap(conf_matrix, xticklabels = class_labels, yticklabels = class_labels, annot = True, cmap="viridis" ,fmt ="g");
        ax.set_ylim([0,9])
        plt.title("CNN Skin Disease Classification Confusion matrix") 
        plt.ylabel('True class') 
        plt.xlabel('Predicted class') 
        plt.show()
        return render(request, 'ViewOutput.html', context)       


def index(request):
    if request.method == 'GET':
       return render(request, 'index.html', {})

def Login(request):
    if request.method == 'GET':
       return render(request, 'Login.html', {})

def Register(request):
    if request.method == 'GET':
       return render(request, 'Register.html', {})   


def Signup(request):
    if request.method == 'POST':
        username = request.POST.get('username', False)
        password = request.POST.get('password', False)
        contact = request.POST.get('contact', False)
        email = request.POST.get('email', False)
        address = request.POST.get('address', False)
        output = "none"
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'skindisease',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select username FROM register")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == username:
                    output = username+" Username already exists"
                    break
        if output == "none":
            db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'skindisease',charset='utf8')
            db_cursor = db_connection.cursor()
            student_sql_query = "INSERT INTO register(username,password,contact,email,address) VALUES('"+username+"','"+password+"','"+contact+"','"+email+"','"+address+"')"
            db_cursor.execute(student_sql_query)
            db_connection.commit()
            print(db_cursor.rowcount, "Record Inserted")
            if db_cursor.rowcount == 1:
                output = "success"
        if output == "success":
            context= {'data':'Signup Process Completed'}
            return render(request, 'Register.html', context)
        else:
            context= {'data':'Username already exists'}
            return render(request, 'Register.html', context)    
        
def UserLogin(request):
    if request.method == 'POST':
        username = request.POST.get('username', False)
        password = request.POST.get('password', False)
        index = 0
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'skindisease',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select * FROM register")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == username and row[1] == password:
                    index = 1
                    break
        if index == 1:
            context= {'data':'welcome '+username}
            return render(request, 'UserScreen.html', context)
        if index == 0:
            context= {'data':'Invalid login details'}
            return render(request, 'Login.html', context)        
        
        
