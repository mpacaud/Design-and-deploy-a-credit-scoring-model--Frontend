## Scoring model implementation: Frontend

### Description

This part of the application is designed to allow the user to interact with the model and the features' interpreter.


### Files and folders

This part of the project's application is splitted within many sub-parts:
1. the main application code can be found in the root folder at dashboard_streamlit.py.
2. Procefile is the starter commands file which allows heroku to launch the dashboard and and setup.sh allows to configure streamlit to allow its run within the heroku environment.
3. steup.sh: Required to proper launch streamlit on heroku.
4. In requirements.txt you can find all required packages.
5. In the Exports folder are stored all required data to import into the app for its functioning.
6. shared_functions.py: Python function shared across many files in the prject.
7. urls.json is a small file which allows to easily switch the address to look for the backend service. In our case, its local address or its online one.


### Online host address

https://mpacaud-oc-ds-p7-app-frontend.herokuapp.com


### Further notes

- For reasons of server resource management, it is possible that the application is not permanently maintained on the Heroku hosting site.
- For general explanations over the whole project got to: https://github.com/mpacaud/Scoring_model_implementation-Development.git