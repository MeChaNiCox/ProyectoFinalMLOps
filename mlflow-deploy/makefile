.PHONY: install train test ui clean

install:
\tpython -m pip install --upgrade pip
\tpip install -r requirements.txt

train:
\tpython src/train.py --config config.yaml

test:
\tpython src/validate.py --metrics artifacts/metrics.json --threshold 0.65

ui:
\tmlflow ui --backend-store-uri file:./mlruns -p 5000

clean:
\trm -rf artifacts __pycache__ .pytest_cache
