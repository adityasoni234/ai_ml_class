from flask import Flask
app = Flask(__name__)

@app.route('/demo')
def hello_world ():
    a = demo()
    return a + '<b>Hello World<b>'

def demo():
    print("Hello World")

if __name__ == '__main__':
    app.run()