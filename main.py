import argparse
import sys
import os
from pydantic import BaseModel
from ocr import OCR_Model
from typing import List, Optional, Dict, Any
from nlp import NLP_Model

# 定义结构化输出的pydantic模型
class OCRResult(BaseModel):
    raw_text: str
    confidence: float = 0.0
    skew_angle: float = 0.0
    entities: List[str] = []

def main():
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description='使用pytesseract进行OCR处理的工具')
    parser.add_argument('input_path', help='要进行OCR处理的文件路径（图像或PDF）')
    parser.add_argument('-o', '--output', help='将结果保存到指定文件', default=None)
    parser.add_argument('--mode', choices=['Public', 'Commercial'], default='Public', 
                       help='OCR模式：Public（使用Tesseract）或Commercial（商业OCR服务，未实现）')
    parser.add_argument('--no-cleaning', action='store_true', help='跳过spaCy文本清洗')
    parser.add_argument('--no-entities', action='store_true', help='跳过实体提取')
    parser.add_argument('--no-sentences', action='store_true', help='跳过句子分割')
    parser.add_argument('--no-confidence', action='store_true', help='跳过置信度计算')
    parser.add_argument('--custom-dict', help='自定义字典文件路径（可选）')
    parser.add_argument('--verbose', action='store_true', help='显示详细分析信息')
    parser.add_argument('--pdf-images-only', action='store_true', 
                       help='仅处理PDF中的图像，跳过直接文本提取')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input_path):
        print(f"错误：文件 '{args.input_path}' 不存在")
        return
    
    # 检查文件类型
    file_ext = os.path.splitext(args.input_path)[1].lower()
    print(f"检测到的文件类型: {file_ext}")


    OCR = OCR_Model()
    NLP = NLP_Model()
    
    # 根据文件类型选择OCR方法
    if file_ext == '.pdf':
        print(f"检测到PDF文件，使用PDF OCR处理...")
        ocr_result = OCR.ocr_pdf(args.input_path, mode=args.mode, custom_dict_path=args.custom_dict)
        
        # 处理PDF OCR结果
        if "error" in ocr_result:
            print(ocr_result["error"])
            return
        
        extracted_text = ocr_result["text"]
        confidence = ocr_result["overall_confidence"]
        skew_angle = ocr_result.get("skew_angle", 0.0)
        
        # 对于PDF，我们可以选择是否提取实体
        if not args.no_entities:
            extracted_entities = NLP.extract_entities(extracted_text)
            
            # 将字典中的实体文本提取出来，合并到一个字符串列表中
            entities_list = []
            for category, items in extracted_entities.items():
                for item in items:
                    if isinstance(item, dict) and "text" in item:
                        entities_list.append(item["text"])
                    elif isinstance(item, str):
                        entities_list.append(item)
        else:
            entities_list = []
        
        # 创建结构化输出
        result = {
            "raw_text": extracted_text,
            "confidence": confidence,
            "skew_angle": skew_angle,
            "entities": entities_list,
            "total_pages": ocr_result.get("total_pages", 0),
            "page_confidences": ocr_result.get("page_confidences", [])
        }
        
    else:
        # 假设是图像文件
        print(f"检测到图像文件，使用图像OCR处理...")
        ocr_result = OCR.ocr_image(args.input_path, mode=args.mode, custom_dict_path=args.custom_dict)
        
        # 检查OCR是否成功
        if "error" in ocr_result:
            print(ocr_result["error"])
            return
        
        # 提取文本、置信度和倾斜角度
        extracted_text = ocr_result["text"]
        confidence = ocr_result["confidence"]
        skew_angle = ocr_result.get("skew_angle", 0.0)
        
        # 提取命名实体
        if not args.no_entities:
            extracted_entities = NLP.extract_entities(extracted_text)
            
            # 将字典中的实体文本提取出来，合并到一个字符串列表中
            entities_list = []
            for category, items in extracted_entities.items():
                for item in items:
                    if isinstance(item, dict) and "text" in item:
                        entities_list.append(item["text"])
                    elif isinstance(item, str):
                        entities_list.append(item)
        else:
            entities_list = []
        
        # 创建结构化输出
        result = OCRResult(
            raw_text=extracted_text,
            confidence=confidence,
            skew_angle=skew_angle,
            entities=entities_list,
        )
    
    # 输出结果
    output_path = args.output
    
    # 如果没有指定输出文件，则默认保存到output文件夹
    if not output_path:
        # 获取输入文件名（不含扩展名）
        input_filename = os.path.splitext(os.path.basename(args.input_path))[0]
        # 构造输出文件路径
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)  # 确保output文件夹存在
        output_path = os.path.join(output_dir, f"{input_filename}.json")
    
    try:
        # 根据结果类型选择保存方式
        if isinstance(result, dict):
            with open(output_path, 'w', encoding='utf-8') as f:
                import json
                json.dump(result, f, ensure_ascii=False, indent=2)
        else:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result.model_dump_json(indent=2))
        
        print(f"OCR结果已保存到 {output_path}")
        
        # 如果是verbose模式，打印更多详细信息
        if args.verbose:
            print("\n=== 详细结果 ===")
            print(f"输入文件: {args.input_path}")
            print(f"处理模式: {args.mode}")
            print(f"置信度: {confidence:.2f}%")
            print(f"倾斜角度: {skew_angle:.2f}度")
            
            if file_ext == '.pdf':
                print(f"总页数: {ocr_result.get('total_pages', 0)}")
                print("\n每页置信度:")
                for page_info in ocr_result.get('page_confidences', []):
                    print(f"  第{page_info['page_number']}页: {page_info['confidence']:.2f}%")
            
            if entities_list:
                print(f"\n提取的实体 ({len(entities_list)}个):")
                for entity in entities_list:
                    print(f"  - {entity}")
    except Exception as e:
        print(f"保存文件时发生错误: {str(e)}")

if __name__ == "__main__":
    main()
