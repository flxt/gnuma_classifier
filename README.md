# gnuma_distilbert
classifier microservice for gnuma project

Before Running the code
1) install the required python packages specified in the requirements.txt
2) Remove the .example from config.json.example and enter the needed values.

Run with 
	python -m src.server path port

where path is the directory name of the calssifier you want to use
and port is the port number you want

Example: 
	python -m src.server distilBERT 4793

To add a new Classifier:
1) Make a copy of the template folder
2) Name it after the Classifier. This directory name will be the path variable if you run the server. e.g. "distilBERT"
3) In the model.json of the directory change the model value to the name of the huggingface model you want. e.g. "distilbert-base-uncased"
4) In the startup.json of the directory change the classifier_name value to the name you want your classifier to have. e.g. "distilbert"
5) If you want you can change the default values in the startup.json. Adding new values is not possible for now.
6) Now you can start the new Classifier as specified above.