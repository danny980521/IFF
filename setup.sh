pip install -r requirements.txt

cd lm-evaluation-harness
pip install -e .[vllm,ifeval]
cd ..

cd FastChat
pip3 install -e ".[model_worker,webui]"

