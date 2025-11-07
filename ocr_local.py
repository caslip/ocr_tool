import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
import os
import io
from PyPDF2 import PdfReader
from typing import Optional, Tuple
from skimage.transform import rotate
from skimage import exposure
import datetime
from langchain_core.tools import tool

pytesseract.pytesseract.tesseract_cmd = r"D:\Program Files\Tesseract-OCR\tesseract.exe"


def ocr_image(image_path: str, mode: str = "Public", custom_dict_path: Optional[str] = None) -> dict:
    """
    使用pytesseract对图像进行OCR处理
    
    参数:
        mode (str): Public:使用tesseract进行OCR; Commercial:使用商业OCR服务（未实现）
        image_path (str): 图像文件路径
        custom_dict_path (str): 自定义字典文件路径（可选）
        
    返回:
        dict: 包含提取的文本和平均置信度的字典
    """
    try:
        if mode == "Commercial":
            return {"error": "商业OCR服务未实现", "text": "", "confidence": 0.0}
        elif mode == "Public":
            model = pytesseract

        # 图像预处理
        # processed_image = preprocess_image(image_path, custom_dict_path)
        
        # 准备Tesseract配置
        config = '--oem 3 --psm 6'
        
        # 如果提供了自定义字典，添加到配置中
        if custom_dict_path and os.path.exists(custom_dict_path):
            config += f' --user-words "{custom_dict_path}"'
        
        # 使用pytesseract进行OCR识别，获取置信度
        lang_param = 'chi_sim+eng'
        
        data = model.image_to_data(image_path, lang=lang_param, config=config, 
                                        output_type=model.Output.DICT)
        
        # 提取文本
        text = ""
        confidence_sum = 0
        confidence_count = 0
        
        for i in range(len(data['text'])):
            if int(data['conf'][i]) > 0:  # 忽略置信度为0的字符
                text += data['text'][i]
                if data['text'][i].strip():  # 只计算非空白字符的置信度
                    confidence_sum += int(data['conf'][i])
                    confidence_count += 1
        
        # 计算平均置信度
        avg_confidence = confidence_sum / confidence_count if confidence_count > 0 else 0.0
        
        # 保存OCR文本到raw_data目录
        try:
            # 确保raw_data目录存在
            raw_data_dir = "raw_data"
            os.makedirs(raw_data_dir, exist_ok=True)
            
            # 生成唯一的文件名，使用时间戳和原始文件名
            original_filename = os.path.splitext(os.path.basename(image_path))[0]
            raw_filename = f"{original_filename}_ocr.txt"
            raw_filepath = os.path.join(raw_data_dir, raw_filename)
            
            # 写入文件
            with open(raw_filepath, 'w', encoding='utf-8') as f:
                f.write(text)
            
            print(f"OCR文本已保存到: {raw_filepath}")
        except Exception as e_save:
            print(f"保存OCR文本时发生错误: {str(e_save)}")
        
        return {"text": text, "confidence": avg_confidence, "skew_angle": detect_skew(cv2.imread(image_path))}
    except Exception as e:
        return {"error": f"OCR处理时发生错误: {str(e)}", "text": "", "confidence": 0.0, "skew_angle": 0.0}

def ocr_pdf(pdf_path: str, mode: str = "Public", custom_dict_path: Optional[str] = None) -> dict:
    """
    对PDF文件中的每一页进行OCR处理
    
    参数:
        mode (str): Public:使用tesseract进行OCR; Commercial:使用商业OCR服务（未实现）
        pdf_path (str): PDF文件路径
        custom_dict_path (str): 自定义字典文件路径（可选）
        
    返回:
        dict: 包含提取的文本、每页置信度和平均置信度的字典
    """
    
    doc = ""
    page_confidences = []
    total_confidence_sum = 0
    total_confidence_count = 0

    # 打开PDF文件
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)

    # 遍历PDF中的每一页
    for page_number, page in enumerate(reader.pages):
        page_text = ""
        page_confidence = 0.0
        page_confidence_count = 0
        
        # 尝试提取页面文本
        text = page.extract_text()
        if text:
            # 如果能直接提取到文本，就添加到Word文档中
            page_text = text
            # 对于直接提取的文本，我们假设置信度为100%
            page_confidence = 100.0
            page_confidence_count = 1
        else:
            # 如果页面没有文本，尝试使用OCR提取图像中的文本
            images = page.images
            if images:
                for image_index, img in enumerate(images):
                    # 将图像数据从PDF中提取出来
                    image = Image.open(io.BytesIO(img.data))
                    
                    # 使用OCR识别图像中的文本，获取置信度
                    data = pytesseract.image_to_data(image, lang='chi_sim', output_type=pytesseract.Output.DICT)
                    
                    page_text_part = ""
                    confidence_sum = 0
                    confidence_count = 0
                    
                    for i in range(len(data['text'])):
                        if int(data['conf'][i]) > 0:  # 忽略置信度为0的字符
                            page_text_part += data['text'][i]
                            if data['text'][i].strip():  # 只计算非空白字符的置信度
                                confidence_sum += int(data['conf'][i])
                                confidence_count += 1
                    
                    # 计算当前图像的平均置信度
                    image_confidence = confidence_sum / confidence_count if confidence_count > 0 else 0.0
                    
                    # 更新页面文本和置信度
                    page_text += (f"第{page_number+1}页, 图像{image_index+1}: {page_text_part}")
                    
                    # 累加到页面置信度
                    if confidence_count > 0:
                        total_confidence_sum += confidence_sum
                        total_confidence_count += confidence_count
                        page_confidence_count += confidence_count
                        page_confidence = total_confidence_sum / total_confidence_count if total_confidence_count > 0 else 0.0
            else:
                # 如果页面既没有文本也没有图像，添加一个占位符
                page_text = (f"第{page_number+1}页无文本或图像。")
                page_confidence = 0.0
                page_confidence_count = 1
        
        # 添加页面文本到文档
        doc += page_text + "\n"
        
        # 记录页面置信度
        page_confidences.append({
            "page_number": page_number + 1,
            "confidence": page_confidence,
            "text_length": len(page_text)
        })
    
    # 计算整体平均置信度
    overall_confidence = total_confidence_sum / total_confidence_count if total_confidence_count > 0 else 0.0
    
    # 保存OCR文本到raw_data目录
    try:
        # 确保raw_data目录存在
        raw_data_dir = "raw_data"
        os.makedirs(raw_data_dir, exist_ok=True)
        
        original_filename = os.path.splitext(os.path.basename(pdf_path))[0]
        raw_filename = f"{original_filename}_ocr.txt"
        raw_filepath = os.path.join(raw_data_dir, raw_filename)
        
        # 写入文件
        with open(raw_filepath, 'w', encoding='utf-8') as f:
            f.write(doc)
        
        print(f"OCR文本已保存到: {raw_filepath}")
    except Exception as e_save:
        print(f"保存OCR文本时发生错误: {str(e_save)}")
    
    # 返回结果
    result = {
        "text": doc,
        "total_pages": total_pages,
        "page_confidences": page_confidences,
        "overall_confidence": overall_confidence,
        "skew_angle": 0.0  # PDF处理不计算倾斜角度
    }
    return result

def detect_skew(image: np.ndarray) -> float:
    """
    检测图像的倾斜角度
    
    参数:
        image (np.ndarray): 输入图像
        
    返回:
        float: 检测到的倾斜角度（度）
    """
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 边缘检测
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # 霍夫变换检测直线
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    
    angles = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            angles.append(angle)
    
    # 计算平均角度
    if angles:
        median_angle = float(np.median(angles))
        # 只考虑-30到30度之间的角度
        if -30 <= median_angle <= 30:
            return median_angle
    
    return 0.0