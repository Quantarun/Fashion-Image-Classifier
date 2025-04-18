__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import pandas as pd
import os
import time
import traceback
from PIL import Image
import requests
from io import BytesIO
from praisonaiagents import Agent, Task, PraisonAIAgents
import tempfile

# Set page config
st.set_page_config(
    page_title="Clothing Classification Agent",
    page_icon="👕",
    layout="wide"
)

# Sidebar for API key input and configuration
with st.sidebar:
    st.title("Configuration")
    api_key = st.text_input("Enter OpenAI API Key", type="password")
    os.environ["OPENAI_API_KEY"] = api_key
    
    st.markdown("---")
    st.subheader("About")
    st.markdown("""
    This app classifies clothing items from images using AI.
    
    Upload an Excel file with image URLs and get classifications for:
    - Specific style
    - Broader category
    """)

# Function to classify a single image
def classify_image(image_url, api_key):
    # Clean up URL (remove any trailing spaces)
    image_url = image_url.strip()
    
    if not api_key:
        return {"Items Style": "API_KEY_MISSING", "Class Broader Category": "API_KEY_MISSING"}
    
    # Create Vision Analysis Agent
    vision_agent = Agent(
        name="ClothingClassifier",
        role="Fashion Product Classifier",
        goal="Accurately classify clothing items into specific categories",
        backstory="""You are an expert in fashion product classification.
        You can identify specific styles and categories of clothing items from images.""",
        llm="gpt-4o",
        self_reflect=False
    )

    # Create classification task with explicit instructions
    classification_task = Task(
        name="Product_Classification",
        description="""Analyze the fashion product in the image.

First, identify the specific style from these options:
- Tops & Blouses: TANK TOP, TEE SHIRT, CROP TOP, CLASSIC SHIRT, BOYFRIEND SHIRT, CROPPED SHIRT, SWEATSHIRT, SHEER BLOUSE, ELEGANT BLOUSE, FASHION BLOUSE
- Bottoms: TROUSERS, TAILORED SHORTS, TAILORED SKIRT, MINI SKIRT, MINI SKIRT WITH LOGO, FASHION MID LENGTH TO LONG SKIRT, LEGGINGS, SWEATPANTS
- Dresses & Jumpsuits: EVENING GOWN, LOGO DRESS, MINI DAY DRESS, DAY TO NIGHT ELEGANT DRESS, LONG DRESS WITH SLIT, JUMPSUIT
- Denim: DENIM SHORTS, DENIM PANTS, DENIM PANTS WITH LOGO, DENIM JACKET, DENIM JACKET WITH LOGO, NOVELTY DENIM
- Outerwear & Jackets: FASHION JACKET, TAILORING JACKET, LEATHER JACKET, FASHION TRENCH, ELEGANT COAT, FASHION COAT, TIMELESS PUFFER, ELEGANT PUFFER
- Shoes & Footwear: SNEAKERS, SLIDES, SEXY BOOTS, ELEGANT BOOTS, CHUNKY BOOTS, OTHER BOOTS, FASHION LOW HEEL, HEELS WITH EMBELLISHMENT/PVC, TIMELESS HEELS, NOVELTY FLAT SANDALS
- Loungewear & Knitwear: EVERYDAY KNIT, FASHION KNITS, LOGO KNIT, DRESSY KNIT, FASHION PUFFER
- Swim & Intimates: BRA, LOGO SWIM, MEDIUM TO LOW COVERAGE, HIGH COVERAGE, TOP OR BODYSUIT WITH LOGO, BODYSUIT
- Party & Statement Pieces: MINI OR PARTY DRESS, DARING AND PARTY PANTS, MINI SKIRT WITH LOGO, BOHEMIAN/BEACHY DRESSES, DIRECTIONAL

understand the image properly identify the image correctly, look for any logos, and then classify the image into the correct style and broader category. Do not include any other text, explanations, or JSON formatting. Only provide the style and category in the specified format.

Next, select the broader category:
Tops & Blouses, Bottoms, Dresses & Jumpsuits, Denim, Outerwear & Jackets, Shoes & Footwear, Loungewear & Knitwear, Swim & Intimates, Party & Statement Pieces

Your response MUST BE ONLY a simple text in this exact format:
Items Style Class: [THE SPECIFIC STYLE]
Broader Category: [THE BROADER CATEGORY]

Do not include any other text, explanations, or JSON formatting.
""",
        expected_output="Simple text with style and category",
        agent=vision_agent,
        images=[image_url]
    )

    try:
        # Create instance and run task
        agents = PraisonAIAgents(
            agents=[vision_agent],
            tasks=[classification_task],
            process="sequential",
            verbose=0  # Reduce verbosity
        )

        # Run the classification
        result = agents.start()
        
        # Initialize default values
        style = "UNKNOWN"
        category = "UNKNOWN"
        
        # Handle the result
        if isinstance(result, str):
            raw_text = result
        elif isinstance(result, dict) and "task_results" in result:
            # Extract the raw result from task_results
            for task_id, task_result in result["task_results"].items():
                if task_result and hasattr(task_result, 'raw'):
                    raw_text = str(task_result.raw)
                    break
                elif task_result and isinstance(task_result, dict) and 'raw' in task_result:
                    raw_text = str(task_result['raw'])
                    break
                else:
                    raw_text = str(task_result)
                    break
        else:
            raw_text = str(result)
        
        # Parse the raw text line by line
        lines = raw_text.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith("Items Style Class:"):
                style = line[len("Items Style Class:"):].strip()
            elif line.startswith("Broader Category:"):
                category = line[len("Broader Category:"):].strip()
        
        return {"Items Style": style, "Class Broader Category": category}
    
    except Exception as e:
        st.error(f"Error during classification: {str(e)}")
        return {"Items Style": "CLASSIFICATION_ERROR", "Class Broader Category": "CLASSIFICATION_ERROR"}

# Function to get image display
def get_image_display(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            return img
        else:
            return None
    except Exception as e:
        st.warning(f"Error loading image: {str(e)}")
        return None

# Main app interface
st.title("👕 Clothing Image Classifier")

# File uploader
uploaded_file = st.file_uploader("Upload Excel file with clothing images", type=['xlsx', 'xls'])

if uploaded_file:
    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_file_path = tmp_file.name
    
    # Load the Excel file
    df = pd.read_excel(temp_file_path)
    
    # Check if required column exists
    if 'Items Image URL' not in df.columns:
        st.error("Error: 'Items Image URL' column not found in the Excel file")
    else:
        # Display the dataframe
        st.subheader("Uploaded Data")
        st.dataframe(df)
        
        # Check if columns already exist, otherwise initialize them
        if 'Items Style' not in df.columns:
            df['Items Style'] = ""
        
        if 'Class Broader Category' not in df.columns:
            df['Class Broader Category'] = ""
        
        # Process button
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Get starting index
            start_index = st.number_input("Starting Index", min_value=0, max_value=len(df)-1, value=0, step=1)
            
            # Process button
            process_button = st.button("Start Processing")
        
        with col2:
            st.info("Click 'Start Processing' to begin classifying images. This may take some time depending on the number of images.")
        
        # Results section
        if process_button:
            if not api_key:
                st.error("Please enter your OpenAI API key in the sidebar before processing.")
            else:
                st.subheader("Processing Results")
                
                # Create a progress bar
                progress_bar = st.progress(0)
                
                # Results container
                results_container = st.empty()
                
                # Process items one by one from the given start index
                for i in range(start_index, len(df)):
                    # Display current item being processed
                    st.write(f"Processing item {i+1}/{len(df)}")
                    
                    # Get the image URL
                    image_url = df.iloc[i]['Items Image URL'] if 'Items Image URL' in df.columns else None
                    
                    # Skip if URL is empty
                    if pd.isna(image_url) or not image_url:
                        st.warning(f"Skipping item {i+1}/{len(df)}: Empty URL")
                        continue
                    
                    # Skip if already processed successfully
                    if pd.notna(df.iloc[i]['Items Style']) and pd.notna(df.iloc[i]['Class Broader Category']) and \
                      df.iloc[i]['Items Style'] not in ["", "UNKNOWN", "ERROR", "NO_RESULT", "DOWNLOAD_ERROR", "JSON_ERROR", "CLASSIFICATION_ERROR", "PARSING_ERROR", "FORMAT_ERROR"]:
                        st.info(f"Skipping item {i+1}/{len(df)}: Already processed - {df.iloc[i]['Items Style']} / {df.iloc[i]['Class Broader Category']}")
                        continue
                    
                    # Display vendor name if available
                    vendor_name = df.iloc[i]['Items Vendor Name'] if 'Items Vendor Name' in df.columns else f"Item {i+1}"
                    
                    # Create columns for this item
                    img_col, info_col = st.columns([1, 2])
                    
                    with img_col:
                        # Display the image
                        img = get_image_display(image_url)
                        if img:
                            st.image(img, caption=vendor_name, width=200)
                        else:
                            st.warning("Image could not be loaded")
                    
                    with info_col:
                        # Classify the image
                        with st.spinner(f"Classifying {vendor_name}..."):
                            result = classify_image(image_url, api_key)
                        
                        # Update the dataframe with result
                        if result:
                            df.at[i, 'Items Style'] = result['Items Style']
                            df.at[i, 'Class Broader Category'] = result['Class Broader Category']
                            
                            # Display the result
                            st.success(f"Classification complete")
                            st.write(f"**Items Style Class:** {result['Items Style']}")
                            st.write(f"**Broader Category:** {result['Class Broader Category']}")
                    
                    # Add a separator
                    st.markdown("---")
                    
                    # Update progress bar
                    progress_bar.progress((i - start_index + 1) / (len(df) - start_index))
                
                # Provide download link for processed data
                st.subheader("Download Processed Data")
                
                # Save to a temporary Excel file
                temp_result_path = os.path.join(tempfile.gettempdir(), "processed_results.xlsx")
                df.to_excel(temp_result_path, index=False)
                
                # Read the file as bytes for download
                with open(temp_result_path, "rb") as file:
                    st.download_button(
                        label="Download Processed Excel",
                        data=file,
                        file_name="clothing_items_classified.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                # Final message
                st.success(f"Processing complete! Processed {len(df) - start_index} items.")
                
                # Display the final results table
                st.subheader("Classification Results")
                st.dataframe(df)
    
    # Clean up the temporary file
    try:
        os.unlink(temp_file_path)
    except:
        pass
else:
    # Display instructions when no file is uploaded
    st.info("Please upload an Excel file with a column named 'Items Image URL' containing URLs to clothing images.")
    
    # Example of how the Excel should be structured
    st.subheader("Example Excel Structure")
    
    example_df = pd.DataFrame({
        "Items Image URL": ["https://example.com/image1.jpg", "https://example.com/image2.jpg"],
        "Items Vendor Name": ["Vendor A", "Vendor B"],
        "Items Style": ["", ""],
        "Class Broader Category": ["", ""]
    })
    
    st.dataframe(example_df)
