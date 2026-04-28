from flask import Flask
import os
from pathlib import Path



def main():
    data_path = r'/mnt/data'
    #data_path = os.getcwd()

    if not os.path.exists(os.path.join(data_path,'upload')):
        os.mkdir(os.path.join(data_path,'upload'))
        
    if not os.path.exists(os.path.join(data_path,'download')):
        os.mkdir(os.path.join(data_path,'download'))

    UPLOAD_FOLDER = Path(os.path.join(data_path,'upload'))
    DOWNLOAD_FOLDER = Path(os.path.join(data_path,'download'))

    app = Flask(__name__)
    app.secret_key = "secret key"
    app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
    app.config['DOWNLOAD_FOLDER'] = str(DOWNLOAD_FOLDER)
    app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024


if __name__ == "__main__":
    main()