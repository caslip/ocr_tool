import streamlit as st
import json
import os
import pandas as pd
from datetime import datetime
import glob

# Set page configuration
st.set_page_config(
    page_title="OCR Result Viewer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styles
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stJson {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
    }
    .stDataFrame {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
    }
    .stExpander {
        border: 1px solid #ddd;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("📄 OCR Result Viewer")
st.markdown("---")

# Get output directory path
output_dir = "output"

# Sidebar - File selection
st.sidebar.title("📁 File Selection")

# Ensure output directory exists
if not os.path.exists(output_dir):
    st.error(f"Output directory '{output_dir}' does not exist!")
    st.stop()

# Get all json files
json_files = []
for root, dirs, files in os.walk(output_dir):
    for file in files:
        if file.endswith('.json'):
            full_path = os.path.join(root, file)
            # Calculate path relative to output directory
            relative_path = os.path.relpath(full_path, output_dir)
            json_files.append(relative_path)

if not json_files:
    st.warning("No JSON files found in the output directory!")
    st.stop()

# File selection
selected_file = st.sidebar.selectbox(
    "Select JSON file to view:",
    json_files,
    index=0
)

# Build full file path
file_path = os.path.join(output_dir, str(selected_file))

# Read JSON file
try:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
except Exception as e:
    st.error(f"Error reading file: {str(e)}")
    st.stop()

# Main interface
st.header(f"📄 {selected_file}")

# Create tabs
tab1, tab2, tab3 = st.tabs(["📋 Raw Data", "📊 Data Analysis", "✏️ Edit Mode"])

# Raw Data tab
with tab1:
    st.subheader("JSON Raw Data")
    
    # Use expander to show JSON data
    with st.expander("View Complete JSON Data", expanded=True):
        st.json(data)
    
    # Display basic information
    st.subheader("Basic Information")
    
    # Extract basic information
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("OCR Mode", data.get("ocr_mode", "Unknown"))
    
    with col2:
        confidence = data.get("confidence", 0)
        st.metric("Confidence", f"{confidence:.2f}%")
    
    with col3:
        skew_angle = data.get("skew_angle", 0)
        st.metric("Skew Angle", f"{skew_angle:.2f}°")
    
    # Display raw text
    if "raw_text" in data:
        st.subheader("📝 Raw Text")
        st.text_area("OCR recognized raw text", data["raw_text"], height=200, disabled=True)
    
    # Display LLM analysis results
    if "llm_analysis" in data and data["llm_analysis"]:
        st.subheader("🤖 LLM Analysis Results")
        st.json(data["llm_analysis"])
    
    # Display processing status
    if "agent_processing" in data:
        st.subheader("📊 Processing Status")
        processing_info = data["agent_processing"]
        status = processing_info.get("status", "Unknown")
        
        if status == "completed":
            st.success(f"Processing Status: {status}")
            if "output_file" in processing_info:
                st.info(f"Output File: {processing_info['output_file']}")
        elif status == "failed":
            st.error(f"Processing Status: {status}")
            if "temp_file" in processing_info:
                st.warning(f"Temporary File: {processing_info['temp_file']}")
        else:
            st.info(f"Processing Status: {status}")

# Data Analysis tab
with tab2:
    st.subheader("📊 Data Analysis")
    
    # If there are LLM analysis results, display visualization of analyzed data
    if "llm_analysis" in data and data["llm_analysis"]:
        llm_data = data["llm_analysis"]
        
        # If LLM analysis contains entity information
        if "entities" in llm_data:
            st.subheader("🏷️ Entity Recognition")
            entities_df = pd.DataFrame(llm_data["entities"])
            if not entities_df.empty:
                st.dataframe(entities_df.style.background_gradient(cmap='Blues'))
            else:
                st.info("No entity information found")
        
        # If LLM analysis contains keywords
        if "keywords" in llm_data:
            st.subheader("🔑 Keywords")
            keywords_df = pd.DataFrame(llm_data["keywords"])
            if not keywords_df.empty:
                st.dataframe(keywords_df.style.background_gradient(cmap='Greens'))
            else:
                st.info("No keyword information found")
        
        # If LLM analysis contains summary
        if "summary" in llm_data:
            st.subheader("📝 Text Summary")
            st.text_area("Summary Content", llm_data["summary"], height=150)
    
    # Display file information
    st.subheader("📁 File Information")
    file_info = {
        "File Path": selected_file,
        "File Size": f"{os.path.getsize(file_path)} bytes",
        "Modification Time": datetime.fromtimestamp(os.path.getmtime(file_path)).strftime('%Y-%m-%d %H:%M:%S'),
        "Confidence": f"{data.get('confidence', 0):.2f}%",
        "Skew Angle": f"{data.get('skew_angle', 0):.2f}°"
    }
    st.json(file_info)

# Edit Mode tab
with tab3:
    st.subheader("✏️ Edit Mode")
    
    # Create editor
    edited_data = st.text_area(
        "Edit JSON Data:",
        value=json.dumps(data, indent=2, ensure_ascii=False),
        height=400,
        key="json_editor"
    )
    
    # Save button
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("💾 Save Changes", type="primary"):
            try:
                # Parse edited JSON
                new_data = json.loads(edited_data)
                
                # Backup original file
                backup_path = file_path + ".backup"
                with open(backup_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                
                # Save new data
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(new_data, f, indent=2, ensure_ascii=False)
                
                st.success("✅ File saved successfully!")
                st.info(f"Original file backed up to: {backup_path}")
                
                # Rerun page to display updated data
                st.rerun()
                
            except json.JSONDecodeError as e:
                st.error(f"❌ JSON format error: {str(e)}")
            except Exception as e:
                st.error(f"❌ Error saving file: {str(e)}")
    
    with col2:
        if st.button("🔄 Reset"):
            st.rerun()

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "OCR Result Viewer © 2025 | Built with Streamlit"
    "</div>",
    unsafe_allow_html=True
)
