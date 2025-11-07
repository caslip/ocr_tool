import argparse
import sys
import os
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from llm_nlp import LLM_NLPProcessor
import base64
import requests
import json
from copy import deepcopy
from ocr_local import ocr_image, ocr_pdf
import datetime

# Define Pydantic model for structured output
class OCRResult(BaseModel):
    raw_text: str
    confidence: float = 0.0
    skew_angle: float = 0.0
    llm_analysis: Optional[Dict[str, Any]] = None

def encode_image(image_path):
    """Reads a local image as bytes and encodes it into a Base64 string.

    Args:
    - image_path: Local path to the image (supports common formats like png/jpg)

    Returns:
    - A Base64 encoded string, decoded from bytes to str
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def perform_mcp_ocr(image_path):
    """Performs OCR processing using the DMX API.

    Args:
    - image_path: Local path to the image

    Returns:
    - A dictionary containing the OCR processing result
    """
    # API Configuration
    BASE_URL = "https://www.dmxapi.cn/v1/"
    API_ENDPOINT = BASE_URL + "chat/completions"
    API_KEY = "sk-************************************************"  # Please replace with your DMXAPI key
                  
    image_data = encode_image(image_path)
    payload = {
        "model": "DeepSeek-OCR-Free",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_data}"
                        }
                    }
                ]
            }
        ],
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    try:
        response = requests.post(API_ENDPOINT, headers=headers, json=payload, timeout=60)
        response.raise_for_status()  # Check if the request was successful
        resp_json = response.json()
        
        # Extract text content
        ocr_text = resp_json["choices"][0]["message"]["content"]
        
        return {
            "raw_text": ocr_text,
            "confidence": 0.0,  # MCP API does not directly provide confidence
            "skew_angle": 0.0,  # MCP API does not directly provide skew angle
            "source": "mcp",
            "full_response": resp_json
        }
    except requests.exceptions.RequestException as e:
        return {"error": f"Error requesting DMX API: {e}"}
    except KeyError:
        return {"error": "Error parsing DMX API response, response format might be incorrect"}
    except Exception as e:
        return {"error": f"Unknown error processing MCP OCR: {e}"}

def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='OCR tool supporting Tesseract and MCP modes')
    parser.add_argument('input_path', help='Path to the file (image or PDF) or directory to be processed')
    parser.add_argument('-o', '--output', help='Save results to a specified file', default=None)
    parser.add_argument('-r', '--directory', action='store_true', help='If the input path is a directory, batch process all files in the directory')
    parser.add_argument('--mode', choices=['Public', 'Commercial', 'MCP'], default='Public', 
                       help='OCR mode: Public (uses Tesseract), Commercial (commercial OCR service, not implemented), or MCP (uses DMX API)')
    parser.add_argument('--no-llm', action='store_true', help='Skip LLM analysis and perform OCR only')
    parser.add_argument('--custom-dict', help='Path to custom dictionary file (optional)')
    parser.add_argument('--verbose', action='store_true', help='Display detailed analysis information')
    parser.add_argument('--ollama-url', default='http://localhost:11434', help='Ollama service URL')
    parser.add_argument('--ollama-model', default='qwen3:8b', help='Ollama model name')
    
    args = parser.parse_args()
    
    if args.directory:
        # Check if input is a directory or a file
        if os.path.isdir(args.input_path):
            print(f"Directory detected, starting batch processing...")
            # Call directory processing function
            process_directory(
                directory_path=args.input_path,
                mode=args.mode,
                no_llm=args.no_llm,
                custom_dict=args.custom_dict,
                verbose=args.verbose,
                ollama_url=args.ollama_url,
                ollama_model=args.ollama_model
            )
    else:
        # Check if input path exists
        if not os.path.exists(args.input_path):
            print(f"Error: '{args.input_path}' doesn't exist.")
            return
        
        if os.path.isdir(args.input_path):
            print(f"Error: '{args.input_path}' is a directory. Please use --directory flag to process directories.")
            return
        
        # Process single file
        file_ext = os.path.splitext(args.input_path)[1].lower()
        print(f"File format: {file_ext}")

        # Initialize LLMProcessor if not skipping LLM analysis
        if not args.no_llm:
            try:
                LLM = LLM_NLPProcessor()
            except Exception as e:
                print(f"Error initializing LLM processor: {e}")
                print("Skipping LLM analysis, proceeding with OCR only.")
                LLM = None
        else:
            LLM = None
        
        # Perform OCR based on mode and file type
        if args.mode == 'MCP':
            if file_ext == '.pdf':
                print("Error: MCP mode does not support PDF files.")
                return
            if file_ext.lower() not in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']:
                print("Error: MCP mode only support (PNG, JPG, JPEG, GIF, BMP, TIFF)")
                return
            
            print(f"Using mcp service to do OCR...")
            ocr_result = perform_mcp_ocr(args.input_path)
            
            if "error" in ocr_result:
                print(ocr_result["error"])
                return
            
            extracted_text = ocr_result["raw_text"]
            confidence = ocr_result["confidence"]
            skew_angle = ocr_result["skew_angle"]
            
            result = {
                "raw_text": extracted_text,
                "confidence": confidence,
                "skew_angle": skew_angle,
                "llm_analysis": None,
                "ocr_mode": "MCP",
                "full_mcp_response": ocr_result.get("full_response")
            }
            
        else: 
            if file_ext == '.pdf':
                print(f"PDF file detected, using PDF OCR processing...")
                ocr_result = ocr_pdf(args.input_path, mode=args.mode, custom_dict_path=args.custom_dict)
            else:
                print(f"Image file detected, using image OCR processing...")
                # Call ocr_image function from ocr.py
                ocr_result = ocr_image(args.input_path, mode=args.mode, custom_dict_path=args.custom_dict)
            
            if "error" in ocr_result:
                print(ocr_result["error"])
                return
            
            extracted_text = ocr_result["text"]
            confidence = ocr_result.get("overall_confidence", ocr_result.get("confidence", 0.0))
            skew_angle = ocr_result.get("skew_angle", 0.0)
        
            result = {
                "raw_text": extracted_text,
                "confidence": confidence,
                "skew_angle": skew_angle,
                "llm_analysis": None,
                "ocr_mode": args.mode
            }
        
        # If not skipping AI Agent processing, perform AI Agent processing
        if LLM and not args.no_llm:
            print(f"Performing AI Agent processing...")
            
            if args.output:
                base_output_path = os.path.splitext(args.output)[0]
            else:
                input_filename = os.path.splitext(os.path.basename(args.input_path))[0]
                base_output_path = os.path.join("output", input_filename)
            
            # Ensure raw_data directory exists
            raw_data_dir = "raw_data"
            os.makedirs(raw_data_dir, exist_ok=True)

            # Generate unique filename using timestamp and original filename
            input_filename = os.path.splitext(os.path.basename(args.input_path))[0]
            raw_filename = f"{input_filename}_ocr.txt"
            temp_txt_path = os.path.join(raw_data_dir, raw_filename)
            
            if args.output:
                base_output_path = os.path.splitext(args.output)[0]
            else:
                base_output_path = os.path.join("output", input_filename)
            agent_output_path = f"{base_output_path}_agent_analysis.json"
            
            try:
                os.makedirs(os.path.dirname(temp_txt_path), exist_ok=True)
                with open(temp_txt_path, 'w', encoding='utf-8') as f:
                    f.write(extracted_text)
                print(f"OCR results saved to temporary file: {temp_txt_path}")
                
                success = False
                llm_analysis_result = LLM.invoke(temp_txt_path)
                result["llm_analysis"] = llm_analysis_result
                success = True
            except Exception as e:
                print(f"Error saving temporary file or calling LLM processing: {e}")
                success = False
            
            if success:
                print("LLM processing completed")
                result["agent_processing"] = {
                    "status": "completed",
                    "output_file": agent_output_path,
                    "temp_file": temp_txt_path
                }
            else:
                print("LLM processing failed")
                result["agent_processing"] = {
                    "status": "failed",
                    "temp_file": temp_txt_path
                }
        
        _save_and_print_result(result, args.output, args.verbose, args.input_path, confidence, skew_angle, file_ext, result.get("ocr_mode"))

def _save_and_print_result(result, output_path, verbose, input_path, confidence, skew_angle, file_ext, ocr_mode):
    """Saves the result and prints detailed information"""
    if not output_path:
        input_filename = os.path.splitext(os.path.basename(input_path))[0]
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{input_filename}.json")
    
    if output_path and not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"OCR results saved to {output_path}")
        
        if verbose:
            print("\n=== Detailed Results ===")
            print(f"Input file: {input_path}")
            print(f"OCR mode: {ocr_mode}")
            print(f"Confidence: {confidence:.2f}%")
            print(f"Skew angle: {skew_angle:.2f} degrees")
            
            if file_ext == '.pdf' and "total_pages" in result:
                print(f"Total pages: {result['total_pages']}")
                if "page_confidences" in result:
                    print("\nPage-wise confidence:")
                    for page_info in result['page_confidences']:
                        print(f"  Page {page_info['page_number']}: {page_info['confidence']:.2f}%")
            
            if "agent_processing" in result:
                print("\n=== AI Agent Processing Results ===")
                agent_processing = result["agent_processing"]
                print(f"Processing status: {agent_processing.get('status', 'Unknown')}")
                
                if agent_processing.get('status') == 'completed':
                    print(f"Output file: {agent_processing.get('output_file', 'Unknown')}")
                elif agent_processing.get('status') == 'failed':
                    print(f"Processing failed: {agent_processing.get('error', 'Unknown error')}")
                    if agent_processing.get('temp_file'):
                        print(f"Temporary file: {agent_processing.get('temp_file')}")
    except Exception as e:
        print(f"Error saving results: {e}")

def process_directory(directory_path, mode='Public', no_llm=False, custom_dict=None, verbose=False, ollama_url='http://localhost:11434', ollama_model='qwen3:8b'):
    """Processes all image and PDF files in a directory.
    
    Args:
        directory_path (str): Path to the directory to be processed.
        mode (str): OCR mode, 'Public' or 'MCP'.
        no_llm (bool): Whether to skip LLM analysis.
        custom_dict (str): Path to the custom dictionary file.
        verbose (bool): Whether to display detailed information.
        ollama_url (str): Ollama service URL.
        ollama_model (str): Ollama model name.
        
    Returns:
        dict: Processing result statistics.
    """
    # Supported file extensions
    image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']
    pdf_extensions = ['.pdf']
    
    # Statistics
    stats = {
        'total_files': 0,
        'processed_files': 0,
        'failed_files': 0,
        'errors': []
    }
    
    # Ensure output directory exists
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize LLM processor
    if not no_llm:
        try:
            LLM = LLM_NLPProcessor()
        except Exception as e:
            print(f"Error initializing LLM processor: {e}")
            print("Skipping LLM analysis, proceeding with OCR only.")
            LLM = None
    else:
        LLM = None
    
    # Iterate through all files in the directory
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        
        # Skip subdirectories
        if os.path.isdir(file_path):
            continue
            
        file_ext = os.path.splitext(filename)[1].lower()
        
        # Only process supported file types
        if file_ext not in image_extensions + pdf_extensions:
            continue
            
        stats['total_files'] += 1
        print(f"\n正在处理文件: {filename} ({stats['processed_files'] + 1}/{stats['total_files']})")
        
        
        # Set output file path
        output_file_name = os.path.splitext(filename)[0]
        directory_name = os.path.basename(os.path.normpath(directory_path))
        print(f"Output directory name: {directory_name}")
        output_dir = os.path.join("output", directory_name)
        output_path = os.path.join(output_dir, f"{output_file_name}.json")
        
        try:
            # Perform OCR processing
            if mode == 'MCP':
                if file_ext == '.pdf':
                    error_msg = "MCP mode does not support PDF files."
                    print(error_msg)
                    stats['failed_files'] += 1
                    stats['errors'].append(f"{filename}: {error_msg}")
                    continue
                    
                if file_ext.lower() not in image_extensions:
                    error_msg = f"MCP mode only supports (PNG, JPG, JPEG, GIF, BMP, TIFF), got {file_ext}"
                    print(error_msg)
                    stats['failed_files'] += 1
                    stats['errors'].append(f"{filename}: {error_msg}")
                    continue
                
                print(f"Using mcp service to do OCR...")
                ocr_result = perform_mcp_ocr(file_path)
                
                if "error" in ocr_result:
                    print(ocr_result["error"])
                    stats['failed_files'] += 1
                    stats['errors'].append(f"{filename}: {ocr_result['error']}")
                    continue
                
                extracted_text = ocr_result["raw_text"]
                confidence = ocr_result["confidence"]
                skew_angle = ocr_result["skew_angle"]
                
                result = {
                    "raw_text": extracted_text,
                    "confidence": confidence,
                    "skew_angle": skew_angle,
                    "llm_analysis": None,
                    "ocr_mode": "MCP",
                    "full_mcp_response": ocr_result.get("full_response")
                }
                
            else: 
                
                if file_ext == '.pdf':
                    print(f"Detect the pdf file, call ocr_pdf...")
                    ocr_result = ocr_pdf(file_path, mode=mode, custom_dict_path=custom_dict)
                    
                else:
                    print(f"Detect the image file, call ocr_image...")
                    ocr_result = ocr_image(file_path, mode=mode, custom_dict_path=custom_dict)
            

                if "error" in ocr_result:
                    print(ocr_result["error"])
                    stats['failed_files'] += 1
                    stats['errors'].append(f"{filename}: {ocr_result['error']}")
                    continue
                
                extracted_text = ocr_result["text"]
                confidence = ocr_result.get("overall_confidence", ocr_result.get("confidence", 0.0))
                skew_angle = ocr_result.get("skew_angle", 0.0)
                
                result = {
                    "raw_text": extracted_text,
                    "confidence": confidence,
                    "skew_angle": skew_angle,
                    "llm_analysis": None,
                    "ocr_mode": mode
                }
            
            # 如果没有跳过AI Agent处理，则进行AI Agent处理
            if LLM and not no_llm:
                print(f"AI Agent is processing...")
                
            # Ensure raw_data directory exists
                raw_data_dir = "raw_data"
                os.makedirs(raw_data_dir, exist_ok=True)

                # Generate unique filename using timestamp and original filename
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                raw_filename = f"{filename}_ocr_{timestamp}.txt"
                temp_txt_path = os.path.join(raw_data_dir, raw_filename)
                
                agent_output_path = os.path.join(output_dir, f"{output_file_name}_agent_analysis.json")
                
                try:
                    os.makedirs(raw_data_dir, exist_ok=True)
                    with open(temp_txt_path, 'w', encoding='utf-8') as f:
                        f.write(extracted_text)
                    print(f"OCR result is successfully saved at: {temp_txt_path}")
                    
                    success = False
                    llm_analysis_result = LLM.invoke(temp_txt_path)
                    result["llm_analysis"] = llm_analysis_result
                    success = True
                except Exception as e:
                    print(f"Error saving temporary file or calling LLM processing: {e}")
                    success = False
                
                if success:
                    print("LLM processing completed")
                    result["agent_processing"] = {
                        "status": "completed",
                        "output_file": agent_output_path,
                        "temp_file": temp_txt_path
                    }
                else:
                    print("LLM processing failed")
                    result["agent_processing"] = {
                        "status": "failed",
                        "temp_file": temp_txt_path
                    }
            
            # 保存结果
            _save_and_print_result(result, output_path, verbose, file_path, confidence, skew_angle, file_ext, mode)
            stats['processed_files'] += 1
            
        except Exception as e:
            error_msg = f"Dealing with {filename} occur mistakes: {str(e)}"
            print(error_msg)
            stats['failed_files'] += 1
            stats['errors'].append(error_msg)
    
    # 打印统计信息
    print("\n=== 处理完成 ===")
    print(f"File amount: {stats['total_files']}")
    print(f"Pass: {stats['processed_files']}")
    print(f"Fail: {stats['failed_files']}")
    
    if stats['errors']:
        print("\nError Information:")
        for error in stats['errors']:
            print(f"  - {error}")
    
    return stats

if __name__ == "__main__":
    main()
