# ocr_tool
# OCR工具 - 优化专业场景

本项目是一个基于Tesseract OCR的优化工具，支持自定义字典、自动校正倾斜图片和去除水印/噪点干扰，以提升OCR识别准确率。

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

## 安装依赖

```bash
pip install -r requirements.txt
```

## 安装spaCy语言包
python -m spacy download zh_core_web_sm
python -m spacy download en_core_web_sm

## 使用方法

### 基本使用

```bash
python main.py 图片路径
```

### 使用自定义字典

```bash
python main.py 图片路径 --custom-dict custom_dict.txt
```

### 其他选项

```bash
python main.py 图片路径 [选项]
```

选项：
- `-o, --output`: 指定输出文件路径
- `--no-cleaning`: 跳过spaCy文本清洗
- `--no-entities`: 跳过实体提取
- `--no-sentences`: 跳过句子分割
- `--custom-dict`: 指定自定义字典文件路径

### 示例

```bash
# 基本OCR识别
python main.py tests/pictures/ch-0.png

# 使用自定义字典
python main.py tests/pictures/ch-0.png --custom-dict custom_dict.txt

# 保存结果到指定文件
python main.py tests/pictures/ch-0.png -o result.json
```

## 自定义字典格式

自定义字典文件是一个简单的文本文件，每行一个词汇或术语。例如：

```
OCR
人工智能
机器学习
深度学习
神经网络
计算机视觉
自然语言处理
模式识别
特征提取
算法
```

## 测试功能

运行测试脚本，演示所有新功能：

```bash
python test_ocr.py
```

测试脚本将：
1. 测试不使用自定义字典的OCR
2. 测试使用自定义字典的OCR
3. 显示图像预处理效果
4. 检测和校正倾斜图片

## 创建倾斜测试图片

使用以下脚本创建倾斜测试图片：

```bash
python create_skewed_test_image.py
```

这将创建一个15度倾斜的测试图片，用于测试倾斜校正功能。

## 输出结果

OCR处理结果包含以下信息：
- 原始文本
- 清洗后文本
- 置信度
- 倾斜角度
- 命名实体
- 句子分割

结果默认保存为JSON格式到`output`目录。

## 技术实现

### 倾斜检测
- 使用Canny边缘检测
- 霍夫变换检测直线
- 计算倾斜角度中位数

### 倾斜校正
- 使用OpenCV的旋转矩阵
- 支持任意角度旋转
- 保持图像质量

### 去噪增强
- 自适应阈值处理
- 形态学操作
- 非局部均值降噪
- CLAHE对比度增强

### 自定义字典
- 使用Tesseract的`--user-words`参数
- 支持中英文混合术语
- 提高专业术语识别率

## 注意事项

1. 确保Tesseract OCR已正确安装并配置
2. 自定义字典文件路径必须正确
3. 倾斜校正功能对≤30°倾斜的图片效果最佳
4. 去噪功能可能需要根据具体图片调整参数

## 许可证

本项目采用MIT许可证。
