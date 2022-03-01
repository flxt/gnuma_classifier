# gnuma_distilbert
classifier microservice for gnuma project

This code was run on linux with python 3.9
The required packages can be faound in the requirements.txt

Run with 
	python -m src.server path port

where path is the directory name of the calssifier you want to use
and port is the port number you want

Example: 
	python -m src.server distilBERT 4793

To add a new Classifier:
1) Make a copy of the template folder
2) Name it after the Classifier. This directory name will be the path variable if you run the server. e.g. "distilBERT" and the name in the frontend
3) In the model.json of the directory change the model value to the name of the huggingface model you want. e.g. "distilbert-base-uncased"
4) Now you can start the new Classifier as specified above.