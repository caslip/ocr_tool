import os
import json
import argparse
from typing import Dict, Any, List, Optional
from datetime import datetime
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.agents import create_agent


def read_txt_file(file_path: str, max_length: int = 8000) -> str:
    """
    读取文本文件内容
    
    Args:
        file_path: 文本文件路径
        max_length: 最大返回长度，避免上下文过长
        
    Returns:
        文件内容字符串，如果出错返回错误信息
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            return f"错误：文件不存在 - {file_path}"
        
        # 检查文件扩展名
        if not file_path.lower().endswith('.txt'):
            return f"错误：只支持 .txt 文件 - {file_path}"
        
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # 如果内容过长，进行截断
        if len(content) > max_length:
            content = content[:max_length] + f"\n... (文件内容已截断，总长度: {len(content)} 字符)"
        
        return content
        
    except Exception as e:
        return f"读取文件时出错: {str(e)}"

class LLM_NLPProcessor:
    def __init__(self) -> None:

        self.config: RunnableConfig = {"configurable": {"thread_id": "1"}}

        self.llm = ChatOllama(
            model="qwen3:8b",
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0.7,
        )
        self.sys_prompt = (
            "You are a helpful assistant that processes image or pdf and generates structured outputs."
            "Output the results in JSON format as specified."
            "Abstract the key information."
        )

        self.agent = create_agent(
            self.llm,
            system_prompt=self.sys_prompt,
            checkpointer=InMemorySaver(),
        )
        
    def invoke(self, file_path: str):

        file_content = read_txt_file(file_path)

        query = f"""
            请分析以下文件内容并提供总结：

            文件路径: {file_path}
            
            文件内容:
            {file_content}
            
            请基于文件内容提供结构化分析和关键要点。
            """
        
        analysis_result_text = ""
        for step in self.agent.stream(
            {"messages": [{"role": "user", "content": query}]},
            self.config,
        ):
            for update in step.values():
                for message in update.get("messages", []):
                    # Assuming message has content attribute, adjust if necessary
                    if hasattr(message, 'content') and message.content:
                        analysis_result_text += message.content + "\n"
                    message.pretty_print() # Keep the pretty print for console output
                    return analysis_result_text
        
if __name__ == "__main__":
    llm = LLM_NLPProcessor()
    llm.invoke(r".\\tests\\pictures\\ch-0.png")

