An OpenAPI compatible server to run Qwen/Qwen3-VL-30B-A3B-Instruct VLM locally on Macs with MPS/GPU acceleration.

--

Setup:
uv venv --python 3.12
uv sync

How to run:
uvicorn main:app --host 0.0.0.0 --port 8000

Now the model is ready to be used.

You can use https://github.com/open-webui/open-webui as a friendly local GUI to test it.

Disclaimer:
This is a proof of concept code, not meant to be used in production. This initial version needs polishing and refactoring.

Enjoy!
