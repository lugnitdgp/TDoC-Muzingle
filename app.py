from flask import Flask,render_template,request
import recommender_system

app = Flask(__name__)
@app.route('/' , methods = ['GET','POST'])

def home():
    if request.method == 'POST' :
        song = request.form['song']
        year = int(request.form['year'])
        song1 = recommender_system.recommender([{"name":song , "year":year}])
        return render_template("base.html" , song1 = song1)
    return render_template("base.html")

if __name__ == "__main__":
    app.run(debug = True)