"""
Application entry point.

All logic lives in the `measurements` package and is exposed via a Flask Blueprint.
"""

from flask import Flask
from measurements.routes import bp

app = Flask(__name__)
app.register_blueprint(bp)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001)