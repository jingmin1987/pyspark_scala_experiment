export PYTHONPATH="$PROJECT_HOME":$PYTHONPATH
jupyter lab --notebook-dir="$PROJECT_HOME/notebook" --no-browser --ServerApp.token=''
