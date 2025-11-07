# ocr_tool
This project is used to simplify the image or pdf identification by the llm.You can just tell the file_path to the main.py, then the script will generate a document about your files.

## 新增功能

### 1. 自定义字典支持
- 支持加载自定义字典文件（行业术语、特殊符号）
- 使用Tesseract的`--user-words`参数增强识别准确率
- 适用于专业领域的OCR识别需求

### 2. 自动校正倾斜图片
- 自动检测图片倾斜角度（支持≤30°倾斜修正）
- 使用OpenCV和霍夫变换进行倾斜检测
- 自动校正倾斜图片，提高OCR识别率

### 3. 去除水印/噪点干扰
- 自适应阈值处理去除背景噪点
- 形态学操作优化文本区域
- 非局部均值降噪去除水印
- 对比度增强提升文本清晰度

## Install dependencies

```bash
pip install -r requirements.txt
```

## Ollama local llm
ollama download address: https://ollama.com/download

ollama pull qwen3:8b

You can also use another chat model if you like, just change the llm configuration of the agent.

## How to use the script

```bash
python main.py 图片路径
```

## License

This project use MIT license
