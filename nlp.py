import spacy
from spacy.pipeline import EntityRuler
from spacy.tokens import Span
from typing import List, Optional, Dict, Any
from langdetect import detect
import json
import re

def load_safe_patterns(json_path):
    """安全加载包含正则表达式的JSON模式"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    patterns = []
    for pattern in data.get("patterns", []):
        fixed_pattern = pattern.copy()
        
        if isinstance(pattern.get("pattern"), list):
            fixed_pattern["pattern"] = []
            for token in pattern["pattern"]:
                fixed_token = token.copy()
                if "TEXT" in token and "REGEX" in token["TEXT"]:
                    # 编译测试正则表达式是否有效
                    try:
                        regex_str = token["TEXT"]["REGEX"]
                        re.compile(regex_str)
                        fixed_token["TEXT"] = {"REGEX": regex_str}
                    except re.error as e:
                        print(f"警告: 正则表达式错误 '{regex_str}': {e}")
                        continue
                
                fixed_pattern["pattern"].append(fixed_token)
        
        patterns.append(fixed_pattern)
    
    return patterns


def debug_patterns(patterns):
    """调试模式"""
    for i, pattern in enumerate(patterns):
        print(f"模式 {i}: {pattern['label']}")
        
        if "TEXT" in pattern['pattern'] and "REGEX" in pattern['pattern']["TEXT"]:
            regex = pattern['pattern']["TEXT"]["REGEX"]
            print(f"  Token: REGEX = {repr(regex)}")
            try:
                re.compile(regex)
                print("  ✓ 正则表达式有效")
            except Exception as e:
                print(f"  ✗ 正则表达式错误: {e}")

# 在加载后调用

class NLP_Model:
    def __init__(self) -> None:
        self.nlp_en = spacy.load("en_core_web_sm")
        self.nlp_zh = spacy.load("zh_core_web_sm")
        self.nlp = self.nlp_zh
        self.entities_path = "ocr_entities.json"  

        with open(self.entities_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 移除现有的entity_ruler（避免重复）
        if "entity_ruler" in self.nlp.pipe_names:
            self.nlp.remove_pipe("entity_ruler")

        ruler = EntityRuler(self.nlp)
        
        # 添加模式到规则器
        patterns = load_safe_patterns(self.entities_path)
        print(f"加载自定义实体模式，共计 {len(patterns)} 条")
        ruler.add_patterns(patterns)
        # debug_patterns(patterns)

        self.nlp.add_pipe("entity_ruler", config={"overwrite_ents": True}, last=True)

        # 将自定义的ruler实例添加到 pipeline 中
        self.nlp.get_pipe("entity_ruler").add_patterns(patterns)

        print(self.nlp.pipe_names)
        # 定义发票相关标签
        self.invoice_labels = {
            "INVOICE_FIELD", "INVOICE_NUMBER", "INVOICE_DATE", 
            "AMOUNT", "TAX_AMOUNT", "COMPANY_NAME", "TAX_ID", "MONEY1"
        }



    def extract_entities(self, text):
        """
        提取文本中的命名实体（不使用置信度计算）
        """
        # 处理文本
        doc = self.nlp(text)
        
        # 调试：打印所有实体
        print("\n=== 调试：所有实体 ===")
        for ent in doc.ents:
            print(f"文本: '{ent.text}' | 标签: {ent.label_} | 起始位置: {ent.start_char}-{ent.end_char}")
        
        # 分类提取实体
        result = {
            "document_fields": [],  # 文档字段
            "companies": [],        # 公司名称  
            "dates": [],            # 日期
            "amounts": [],          # 金额
            "other_entities": []    # 其他实体
        }
        
        # 提取所有实体
        # 可视化展示
        print("\n实体可视化:")
        print("-" * 50)
        for ent in doc.ents:
            print(f"{ent.text} → {ent.label_}")
        
        # 处理实体分类
        for ent in doc.ents:
            entity_data = {
                "text": ent.text,
                "label": ent.label_,
                "position": [ent.start_char, ent.end_char]
            }
            
            if ent.label_ in ["INVOICE_FIELD"]:
                result["document_fields"].append(entity_data)
            elif ent.label_ == "ORG":
                result["companies"].append(entity_data)
            elif ent.label_ == "DATE":
                result["dates"].append(entity_data)
            elif ent.label_ in ["MONEY", "MONEY1"]:
                result["amounts"].append(entity_data)
            else:
                result["other_entities"].append(entity_data)
        
        
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result

    # def print_entity_analysis(self, text: str) -> None:
    #     """
    #     打印实体分析结果，包括置信度信息
    #     """
    #     doc = self.nlp(text)
        
    #     print("\n=== 实体分析结果 ===")
    #     print("-" * 60)
        
    #     # 处理每个实体
    #     for ent in doc.ents:
    #         # 计算置信度
    #         confidence = self.confidence_calc.calculate_invoice_confidence(
    #             ent.text, ent.label_, text
    #         )
            
    #         # 获取置信度级别
    #         confidence_level = self.confidence_calc.get_confidence_level(confidence)
            
    #         # 设置实体扩展属性
    #         ent._.confidence = confidence
    #         ent._.confidence_level = confidence_level
    #         ent._.needs_verification = confidence < 0.7
    #         ent._.is_invoice_field = ent.label_ in self.invoice_labels
            
    #         # 打印实体信息
    #         verification_status = "✓" if confidence >= 0.7 else "✗需校验"
    #         invoice_field_status = "是" if ent.label_ in self.invoice_labels else "否"
            
    #         print(f"文本: '{ent.text:20} | "
    #               f"类型: {ent.label_:15} | "
    #               f"置信度: {confidence:.2f} | "
    #               f"级别: {confidence_level:6} | "
    #               f"发票字段: {invoice_field_status:3} | "
    #               f"{verification_status}")
        
    #     # 打印置信度摘要
    #     result = self.extract_entities_with_confidence(text)
    #     print("\n=== 置信度摘要 ===")
    #     print("-" * 40)
    #     print(f"总实体数: {result['summary']['total_entities']}")
    #     print(f"高置信度 (≥0.8): {result['summary']['high_confidence']}")
    #     print(f"中等置信度 (0.6-0.79): {result['summary']['medium_confidence']}")
    #     print(f"低置信度 (<0.6): {result['summary']['low_confidence']}")
    #     print(f"需人工校验: {result['summary']['needs_verification']}")
        
    #     # 打印建议
    #     if result['summary']['needs_verification'] > 0:
    #         print("\n⚠️  建议: 以下实体需要人工校验:")
    #         for category in ["document_fields", "companies", "dates", "amounts", "other_entities"]:
    #             for entity in result[category]:
    #                 if entity["needs_verification"]:
    #                     print(f"  - '{entity['text']}' ({entity['label']}, 置信度: {entity['confidence']:.2f})")
    #     else:
    #         print("\n✅ 所有实体置信度良好，无需特别校验。")


if __name__ == "__main__":
    nlp = NLP_Model()
    
    # 测试文本
    test_text = """
    增值税专用发票
    发票号码：NO.AB20231025
    开票日期：2023年10月25日
    购买方：北京某某科技有限公司
    销售方：上海某某股份有限公司
    金额：￥5,000.00
    税率：13%
    税额：￥650.00
    纳税人识别号：91110105MA01XYZ123
    """
    
    # 也可以直接获取结构化结果
    print("\n=== 结构化结果 ===")
    structured_result = nlp.extract_entities(test_text)
    # print(json.dumps(structured_result, ensure_ascii=False, indent=2))
