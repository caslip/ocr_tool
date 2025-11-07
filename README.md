# ocr_tool
This project is used to simplify the image or pdf identification by the llm.You can just tell the file_path to the main.py, then the script will generate a document about your files.

## New Features

### 1. Processing a directory

Now, you can use the command to batch processing your files as follow:

```bash
python main.py -r /your/pdf/ or /your/images/
python main.py --directory /your/pdf/ or /your/images/
```

### 2. Using llm to analyze files
We using the Ollama qwen3:8b to process the ocr_raw_data, it will give you some key information and remind you to check the ocr data and the image data.

You can also using --no-llm to skip the llm processing if you want to see the result of OCR directly.


## Installation
1. Install python libraries

```bash
pip install -r requirements.txt
```

2. Login or sign up the website: https://www.dmxapi.cn/

Apply your own key to instead the API_KEY to use the deepseek_ocr_free model

3. Install ollama
ollama download address: https://ollama.com/download

ollama pull qwen3:8b

You can also use another chat model if you like, just change the llm configuration of the agent which is located in llm_nlp.py

For example, login or sign up the openai website, apply the API key to call the free model, and use langchain to instead the ollama llm.

## How to use the script

```bash
python main.py your_path
```

## Test
python main.py tests/pictures/ch-2.png
python main.py -r tests/pictures
python main.py -r tests/pdf

## License

This project use MIT license
