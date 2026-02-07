from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    # Implement detection logic here
    landing_spot = request.form['landing_spot']
    return f'Detected landing spot: {landing_spot}'

if __name__ == '__main__':
    app.run(debug=True)