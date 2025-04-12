pip install -r requirements.txt

cd lm-evaluation-harness
pip install -e .[ifeval]
cd ..

cd FastChat
pip3 install -e ".[model_worker,webui]"

