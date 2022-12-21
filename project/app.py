from flask import Flask, render_template, request
import recommender_system

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def home():
	if request.method == 'POST':
		song_name = request.form["song"]
		song_year = request.form["year"]
		song_year = int(song_year)
		recommended = recommender_system.recommender([{'name':song_name, 'year': song_year}])
		return render_template("base.html", recommended=recommended)

	return render_template("base.html")

if __name__ == '__main__':
	app.run(debug=True)