from flask import Flask
from __init__ import yt_bp
import sys
import os

# Add ShopInsight Modules folder to import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ShopInsight', 'Modules')))

from shopinsight import shopinsight_bp  # 👈 Import the blueprint

app = Flask(__name__)
app.secret_key = 'dev'
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

app.register_blueprint(yt_bp)
app.register_blueprint(shopinsight_bp)  # 👈 Register new blueprint

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5001)
