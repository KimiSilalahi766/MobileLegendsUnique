import os
from app import app

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))  # default 5000 jika tidak ada env
    app.run(host='0.0.0.0', port=port)
