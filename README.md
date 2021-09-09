# diagnose-diabetes-svm
This project is on diagnosing diabetes in patients based on their medical records using machine learning. The ML algorithm used here is the Support Vector Machine. I used the four SVM kernels in Sci-kit Learn; linear, sigmoid, poly, and rbf.
The article for this project is [here](https://www.section.io/engineering-education/diagnose-diabetes-with-svm

First, install a virtual environment.

```bash
pip install virtualenv
```

We create a virtual environment and call it venv.

```bash
virtualenv venv
```

To activate the virtual environment on Windows, run the command below.

```bash
venv\Scripts\activate.bat 
```

The `requirements.txt` file contains all the libraries you need to install to run the code.
In your terminal, run the code below:

```bash
pip install -r requirements.txt
```

Uncomment the last line of code in the api.py file to run the code locally.

```python
app.run(port=5000, debug=True)
```

Run the code below to use the API:

```bash
python api.py
```

You can test the API with POSTMAN.
