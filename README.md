> How to use:

- Install python 3.10

1) Set up `venv` in python

- $ `pip install virtualenv`
- $ `python -m venv python`
- $ `cd python`
- $ `cd Scripts`
- $ `Activate`
- Then go back to main directory


2) Copy The Models

- Go to the [Model Link](https://drive.google.com/drive/folders/1Bp68sAxfVPjMFU7ita-LnDNu4D_UXcaH?usp=sharing) and
  download the folder `models`
- Place the `models` folder in the project. Folders ai, apis, models and etc should be in the same directory

3) Install requirements
   
   $ `pip install -r requirements.txt`

> How to run:

1) $ `cd APIs`
2) $ `uvicorn source:app --port=8001 --reload`
