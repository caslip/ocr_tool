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

pytesseract.pytesseract.tesseract_cmd = r'D:\Program Files\Tesseract-OCR\tesseract.exe'  # 取消注释并修改为您的Tesseract安装路径


class OCR_Model:
    def __init__(self) -> None:
        pass


    def detect_skew(self, image: np.ndarray) -> float:
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


    def deskew_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """
        校正图像倾斜
        
        参数:
            image (np.ndarray): 输入图像
            angle (float): 倾斜角度
            
        返回:
            np.ndarray: 校正后的图像
        """
        if abs(angle) > 0.1:  # 只有当角度足够大时才进行校正
            # 计算图像中心
            height, width = image.shape[:2]
            center = (width // 2, height // 2)
            
            # 获取旋转矩阵
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # 执行旋转
            rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), 
                                         flags=cv2.INTER_CUBIC, 
                                         borderMode=cv2.BORDER_REPLICATE)
            return rotated_image
        return image


    def remove_watermark_and_noise(self, image: np.ndarray) -> np.ndarray:
        """
        去除水印和噪点
        
        参数:
            image (np.ndarray): 输入图像
            
        返回:
            np.ndarray: 处理后的图像
        """
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 自适应阈值处理
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
        
        # 形态学操作去除噪点
        kernel = np.ones((1, 1), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # 降噪
        denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        
        # 增强对比度
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        return enhanced


    def preprocess_image(self, image_path: str, custom_dict_path: Optional[str] = None) -> Image.Image:
        """
        图像预处理，包括倾斜校正、去噪和增强
        
        参数:
            image_path (str): 图像文件路径
            custom_dict_path (str): 自定义字典文件路径（可选）
            
        返回:
            Image.Image: 处理后的PIL图像
        """
        try:
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"无法读取图像: {image_path}")
            
            # 去除水印和噪点
            cleaned_image = self.remove_watermark_and_noise(image)
            
            # 检测并校正倾斜
            skew_angle = self.detect_skew(cleaned_image)
            deskewed_image = self.deskew_image(cleaned_image, skew_angle)
            
            # 转换为PIL图像
            pil_image = Image.fromarray(cv2.cvtColor(deskewed_image, cv2.COLOR_BGR2RGB))
            
            # 增强文本
            enhancer = ImageEnhance.Contrast(pil_image)
            pil_image = enhancer.enhance(2.0)
            
            # 锐化
            pil_image = pil_image.filter(ImageFilter.SHARPEN)
            
            # 调整大小（如果需要）
            # width, height = pil_image.size
            # if width > 2000 or height > 2000:
            #     ratio = min(2000/width, 2000/height)
            #     new_size = (int(width * ratio), int(height * ratio))
            #     pil_image = pil_image.resize(new_size, Image.LANCZOS)
            
            return pil_image
        except Exception as e:
            print(f"图像预处理时发生错误: {str(e)}")
            # 返回原始图像
            return Image.open(image_path)


    def ocr_image(self, image_path: str, mode: str = "Public", custom_dict_path: Optional[str] = None) -> dict:
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
                self.model = pytesseract

            # 图像预处理
            processed_image = self.preprocess_image(image_path, custom_dict_path)
            
            # 准备Tesseract配置
            config = '--oem 3 --psm 6'
            
            # 如果提供了自定义字典，添加到配置中
            if custom_dict_path and os.path.exists(custom_dict_path):
                config += f' --user-words "{custom_dict_path}"'
            
            # 使用pytesseract进行OCR识别，获取置信度
            lang_param = 'chi_sim+eng'
            
            data = self.model.image_to_data(processed_image, lang=lang_param, config=config, 
                                          output_type=self.model.Output.DICT)
            
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
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                original_filename = os.path.splitext(os.path.basename(image_path))[0]
                raw_filename = f"{original_filename}_ocr_{timestamp}.txt"
                raw_filepath = os.path.join(raw_data_dir, raw_filename)
                
                # 写入文件
                with open(raw_filepath, 'w', encoding='utf-8') as f:
                    f.write(text)
                
                print(f"OCR文本已保存到: {raw_filepath}")
            except Exception as e_save:
                print(f"保存OCR文本时发生错误: {str(e_save)}")
            
            return {"text": text, "confidence": avg_confidence, "skew_angle": self.detect_skew(cv2.imread(image_path))}
        except Exception as e:
            return {"error": f"OCR处理时发生错误: {str(e)}", "text": "", "confidence": 0.0, "skew_angle": 0.0}

    def ocr_pdf(self, pdf_path: str, mode: str = "Public", custom_dict_path: Optional[str] = None) -> dict:
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
            
            # 生成唯一的文件名，使用时间戳和原始文件名
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            original_filename = os.path.splitext(os.path.basename(pdf_path))[0]
            raw_filename = f"{original_filename}_ocr_{timestamp}.txt"
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
        
        # 打印结果（可选）
        print("\n=== PDF OCR 处理结果 ===")
        print(f"总页数: {total_pages}")
        print(f"整体置信度: {overall_confidence:.2f}%")
        print("\n每页置信度:")
        for page_info in page_confidences:
            print(f"第{page_info['page_number']}页: 置信度={page_info['confidence']:.2f}%, 文本长度={page_info['text_length']}")
        
        return result

if __name__ == "__main__":
    ocr_model = OCR_Model()
    # 示例：对单张图像进行OCR
    # result = ocr_model.ocr_image("sample_image.png", mode="Public")
    # print("OCR结果:", result)

    # 示例：对PDF文件进行OCR
    ocr_model.ocr_pdf("tests/pdf/定义需求_扫描版.pdf", mode="Public")
