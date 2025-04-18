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
import tempfile
import logging
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Clothing Classification Agent",
    page_icon="👕",
    layout="wide"
)

# Add CSS for better styling
# st.markdown("""
# <style>
#     .result-box {
#         background-color: #f0f2f6;
#         border-radius: 10px;
#         padding: 20px;
#         margin-bottom: 20px;
#     }
#     .success-text {
#         color: #28a745;
#         font-weight: bold;
#     }
#     .error-text {
#         color: #dc3545;
#         font-weight: bold;
#     }
#     .warning-text {
#         color: #ffc107;
#         font-weight: bold;
#     }
#     .info-text {
#         color: #17a2b8;
#         font-weight: bold;
#     }
#     .api-key-input {
#         margin-bottom: 20px;
#     }
#     .verification-badge {
#         color: #28a745;
#         font-size: 1.2em;
#         margin-left: 10px;
#     }
#     .double-check-result {
#         background-color: #e9f7ef;
#         border-left: 5px solid #28a745;
#         padding: 10px;
#         margin-top: 10px;
#     }
# </style>
# """, unsafe_allow_html=True)

# Function to get session state variables with default values
def get_session_var(key, default=None):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]

# Initialize session state variables
if 'api_key_verified' not in st.session_state:
    st.session_state.api_key_verified = False
if 'classification_results' not in st.session_state:
    st.session_state.classification_results = {}

# Check if OPENAI_API_KEY is already set in environment or session
api_key = os.environ.get("OPENAI_API_KEY", "")
if not api_key:
    api_key = get_session_var('openai_api_key', "")

# Function to get image display
def get_image_display(url):
    try:
        # Add a timeout to prevent hanging on slow connections
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            return img
        else:
            logger.warning(f"Failed to load image, status code: {response.status_code}")
            return None
    except requests.exceptions.Timeout:
        logger.warning(f"Timeout while loading image: {url}")
        return None
    except requests.exceptions.RequestException as e:
        logger.warning(f"Request error loading image: {str(e)}")
        return None
    except Exception as e:
        logger.warning(f"Error loading image: {str(e)}")
        return None

# Function to classify an image using direct OpenAI API
def classify_image(image_url, attempt=1):
    """
    Classify an image using direct OpenAI API call
    """
    try:
        # Import OpenAI here to avoid loading it if not needed
        from openai import OpenAI
        client = OpenAI()
        
        # Check if API key is set
        if not os.environ.get("OPENAI_API_KEY"):
            return {"Items Style": "API_KEY_MISSING", "Class Broader Category": "API_KEY_MISSING"}
        
        # Clean up URL
        image_url = image_url.strip()
        
        # Define the prompt for classification
        # Using different prompts for first and second attempts to get varied perspectives
        if attempt == 1:
            prompt = """
            Analyze the fashion product in the image.

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

            Next, select the broader category:
            Tops & Blouses, Bottoms, Dresses & Jumpsuits, Denim, Outerwear & Jackets, Shoes & Footwear, Loungewear & Knitwear, Swim & Intimates, Party & Statement Pieces

            Your response MUST BE ONLY a simple text in this exact format:
            Items Style Class: [THE SPECIFIC STYLE]
            Broader Category: [THE BROADER CATEGORY]

            Do not include any other text, explanations, or JSON formatting.
            """
        else:
            # Different approach for second attempt to get more robust results
            prompt = """
            Look at this clothing item carefully.

            Examine the details such as fabric, cut, design, and purpose of this fashion item.

            Choose the most appropriate specific style from these categories:
            - Tops & Blouses: TANK TOP, TEE SHIRT, CROP TOP, CLASSIC SHIRT, BOYFRIEND SHIRT, CROPPED SHIRT, SWEATSHIRT, SHEER BLOUSE, ELEGANT BLOUSE, FASHION BLOUSE
            - Bottoms: TROUSERS, TAILORED SHORTS, TAILORED SKIRT, MINI SKIRT, MINI SKIRT WITH LOGO, FASHION MID LENGTH TO LONG SKIRT, LEGGINGS, SWEATPANTS
            - Dresses & Jumpsuits: EVENING GOWN, LOGO DRESS, MINI DAY DRESS, DAY TO NIGHT ELEGANT DRESS, LONG DRESS WITH SLIT, JUMPSUIT
            - Denim: DENIM SHORTS, DENIM PANTS, DENIM PANTS WITH LOGO, DENIM JACKET, DENIM JACKET WITH LOGO, NOVELTY DENIM
            - Outerwear & Jackets: FASHION JACKET, TAILORING JACKET, LEATHER JACKET, FASHION TRENCH, ELEGANT COAT, FASHION COAT, TIMELESS PUFFER, ELEGANT PUFFER
            - Shoes & Footwear: SNEAKERS, SLIDES, SEXY BOOTS, ELEGANT BOOTS, CHUNKY BOOTS, OTHER BOOTS, FASHION LOW HEEL, HEELS WITH EMBELLISHMENT/PVC, TIMELESS HEELS, NOVELTY FLAT SANDALS
            - Loungewear & Knitwear: EVERYDAY KNIT, FASHION KNITS, LOGO KNIT, DRESSY KNIT, FASHION PUFFER
            - Swim & Intimates: BRA, LOGO SWIM, MEDIUM TO LOW COVERAGE, HIGH COVERAGE, TOP OR BODYSUIT WITH LOGO, BODYSUIT
            - Party & Statement Pieces: MINI OR PARTY DRESS, DARING AND PARTY PANTS, MINI SKIRT WITH LOGO, BOHEMIAN/BEACHY DRESSES, DIRECTIONAL

            Then determine which broader category it belongs to:
            Tops & Blouses, Bottoms, Dresses & Jumpsuits, Denim, Outerwear & Jackets, Shoes & Footwear, Loungewear & Knitwear, Swim & Intimates, Party & Statement Pieces

            Respond with ONLY this format:
            Items Style Class: [THE SPECIFIC STYLE]
            Broader Category: [THE BROADER CATEGORY]

            No additional text or explanations.
            """
        
        # Make the API call to OpenAI
        response = client.chat.completions.create(
            model="o4-mini",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a fashion product classification expert with knowledge of retail clothing categories."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ],
            max_completion_tokens=300
        )
        
        # Extract the response
        raw_text = response.choices[0].message.content
        
        # Parse the raw text line by line
        style = "UNKNOWN"
        category = "UNKNOWN"
        
        lines = raw_text.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith("Items Style Class:"):
                style = line[len("Items Style Class:"):].strip()
            elif line.startswith("Broader Category:"):
                category = line[len("Broader Category:"):].strip()
        
        return {"Items Style": style, "Class Broader Category": category}
    
    except Exception as e:
        logger.error(f"Error during classification attempt {attempt}: {str(e)}")
        logger.error(traceback.format_exc())
        return {"Items Style": f"CLASSIFICATION_ERROR_ATTEMPT_{attempt}", "Class Broader Category": f"CLASSIFICATION_ERROR_ATTEMPT_{attempt}"}

# Function to verify classification results
def verify_classifications(result1, result2):
    """
    Compare two classification results and determine the final verdict
    """
    # If both classifications are identical, return the first one (high confidence)
    if result1['Items Style'] == result2['Items Style'] and result1['Class Broader Category'] == result2['Class Broader Category']:
        return result1, True
    
    # If both classify to the same broader category but different styles, take the first one
    # but mark as medium confidence
    if result1['Class Broader Category'] == result2['Class Broader Category']:
        return result1, False
    
    # If completely different, combine both with the first one as primary
    combined_result = {
        "Items Style": result1['Items Style'],
        "Class Broader Category": result1['Class Broader Category'],
        "Alternative Style": result2['Items Style'],
        "Alternative Category": result2['Class Broader Category']
    }
    
    return combined_result, False

# Function to test OpenAI API connection
def test_openai_connection():
    try:
        from openai import OpenAI
        client = OpenAI()
        
        # Try a simple completion to test the connection
        response = client.chat.completions.create(
            model="o4-mini",
            messages=[{"role": "user", "content": "Hello"}],
            max_completion_tokens=5
        )
        return True, "Connection successful"
    except Exception as e:
        return False, str(e)

# Function to set API key
def set_api_key(key):
    # Store in session state 
    st.session_state.openai_api_key = key
    # Set in environment for current session
    os.environ["OPENAI_API_KEY"] = key
    # Test if valid
    success, message = test_openai_connection()
    st.session_state.api_key_verified = success
    return success, message

# Sidebar for API key input and configuration
with st.sidebar:
    st.title("Configuration")
    
    # Improved API key input with verification
    st.subheader("OpenAI API Key")
    
    # Show a secured input field for the API key
    api_key_input = st.text_input(
        "Enter your OpenAI API Key",
        type="password",
        value=api_key,
        key="api_key_input_field",
        help="Your API key will be stored temporarily for this session only"
    )
    
    # Verify button with improved feedback
    if st.button("Verify API Key"):
        if api_key_input:
            with st.spinner("Verifying API key..."):
                success, message = set_api_key(api_key_input)
                if success:
                    st.success(f"✅ API key verified successfully!")
                    st.session_state.api_key_verified = True
                else:
                    st.error(f"❌ API key verification failed: {message}")
                    st.session_state.api_key_verified = False
        else:
            st.warning("Please enter an API key")
    
    # Show verification status
    if st.session_state.api_key_verified:
        st.markdown('<p class="success-text">API key is active and ready to use ✓</p>', unsafe_allow_html=True)
    
    # Add API key troubleshooting section
    # with st.expander("API Key Troubleshooting"):
    #     st.markdown("""
    #     If you're experiencing API key issues:
        
    #     1. **Create a .env file** in the same directory as your script with:
    #        ```
    #        OPENAI_API_KEY=your_key_here
    #        ```
    #     2. **Verify your key** is correct and active in your [OpenAI account](https://platform.openai.com/account/api-keys)
    #     3. **Check billing status** in your OpenAI account
    #     4. **Try a different model** if your key doesn't have access to certain models
    #     """)
    
    # Add classification settings
    st.subheader("Classification Settings")
    
    # Toggle for double-checking images
    double_check = st.toggle("Enable Double-Check", value=True, help="Run classification twice for higher accuracy")
    
    # Add confidence threshold slider
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Minimum confidence level required to accept a classification"
    )

# Main app interface
st.title("👕 Clothing Image Classifier")
st.markdown("### Upload an Excel file with clothing image URLs for automatic classification")

# Add API key check warning
if not api_key or not st.session_state.api_key_verified:
    st.warning("⚠️ Please enter and verify your OpenAI API key in the sidebar before proceeding.")

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
        
        # Add confidence column if double-checking is enabled
        if 'Classification Confidence' not in df.columns:
            df['Classification Confidence'] = ""
        
        # Add alternative classifications columns if needed
        if 'Alternative Style' not in df.columns:
            df['Alternative Style'] = ""
        
        if 'Alternative Category' not in df.columns:
            df['Alternative Category'] = ""
        
        # Process button
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Get starting index
            start_index = st.number_input("Starting Index", min_value=0, max_value=len(df)-1, value=0, step=1)
            
            # Add batch size option
            batch_size = st.number_input("Batch Size", min_value=1, max_value=len(df), value=min(5, len(df)), step=1,
                                         help="Number of items to process before saving results")
            
            # Process button
            process_button = st.button("Start Processing")
        
        with col2:
            st.info("Click 'Start Processing' to begin classifying images. Double-check is enabled to ensure accuracy.")
            
            # Show additional information about the process
            if double_check:
                st.markdown("""
                **Double-Check Mode Enabled**
                - Each image will be classified twice with different prompts
                - Results with matching classifications will have high confidence
                - Results with differences will show alternative classifications
                """)
        
        # Results section
        if process_button:
            if not api_key or not st.session_state.api_key_verified:
                st.error("Please enter and verify your OpenAI API key in the sidebar before processing.")
            else:
                st.subheader("Processing Results")
                
                # Create a progress bar
                progress_bar = st.progress(0)
                
                # Results container
                results_container = st.empty()
                
                # Process items one by one from the given start index
                for i in range(start_index, len(df)):
                    # Get the image URL
                    image_url = df.iloc[i]['Items Image URL'] if 'Items Image URL' in df.columns else None
                    
                    # Skip if URL is empty
                    if pd.isna(image_url) or not image_url:
                        st.warning(f"Skipping item {i+1}/{len(df)}: Empty URL")
                        continue
                    
                    # Skip if already processed successfully with high confidence
                    if (pd.notna(df.iloc[i]['Items Style']) and 
                        pd.notna(df.iloc[i]['Class Broader Category']) and
                        df.iloc[i]['Items Style'] not in ["", "UNKNOWN", "ERROR", "NO_RESULT", "DOWNLOAD_ERROR", "JSON_ERROR", "CLASSIFICATION_ERROR", "PARSING_ERROR", "FORMAT_ERROR"] and
                        str(df.iloc[i]['Classification Confidence']).lower() == "high"):
                        
                        st.info(f"Skipping item {i+1}/{len(df)}: Already processed with high confidence - {df.iloc[i]['Items Style']} / {df.iloc[i]['Class Broader Category']}")
                        continue
                    
                    # Display vendor name if available
                    vendor_name = df.iloc[i]['Items Vendor Name'] if 'Items Vendor Name' in df.columns else f"Item {i+1}"
                    
                    # Create a box for this item using st.container and custom CSS
                    with st.container():
                        st.markdown('<div class="result-box">', unsafe_allow_html=True)
                        
                        # Create columns for this item
                        img_col, info_col = st.columns([1, 2])
                        
                        with img_col:
                            # Display the image
                            img = get_image_display(image_url)
                            if img:
                                st.image(img, caption=vendor_name, width=200)
                            else:
                                st.markdown('<p class="warning-text">Image could not be loaded</p>', unsafe_allow_html=True)
                        
                        with info_col:
                            # Show item details
                            st.markdown(f"**Processing:** Item {i+1}/{len(df)}")
                            if 'Items Vendor Name' in df.columns:
                                st.markdown(f"**Vendor:** {vendor_name}")
                            st.markdown(f"**URL:** [{image_url[:50]}...]({image_url})")
                            
                            # First classification attempt
                            with st.spinner(f"Classifying {vendor_name} (Attempt 1)..."):
                                try:
                                    result1 = classify_image(image_url, attempt=1)
                                    
                                    # Check for errors in the classification result
                                    if "ERROR" in result1.get('Items Style', '') or "MISSING" in result1.get('Items Style', ''):
                                        st.markdown(f'<p class="error-text">First classification attempt failed: {result1["Items Style"]}</p>', unsafe_allow_html=True)
                                    else:
                                        st.markdown('<p class="success-text">First classification complete ✓</p>', unsafe_allow_html=True)
                                        
                                except Exception as e:
                                    st.markdown(f'<p class="error-text">First classification failed: {str(e)}</p>', unsafe_allow_html=True)
                                    result1 = {"Items Style": "CLASSIFICATION_ERROR", "Class Broader Category": "CLASSIFICATION_ERROR"}
                            
                            # Second classification attempt (if double-check is enabled)
                            result2 = None
                            if double_check and "ERROR" not in result1.get('Items Style', '') and "MISSING" not in result1.get('Items Style', ''):
                                with st.spinner(f"Double-checking {vendor_name} (Attempt 2)..."):
                                    try:
                                        result2 = classify_image(image_url, attempt=2)
                                        
                                        # Check for errors in the second classification result
                                        if "ERROR" in result2.get('Items Style', '') or "MISSING" in result2.get('Items Style', ''):
                                            st.markdown(f'<p class="warning-text">Second classification attempt failed: {result2["Items Style"]}</p>', unsafe_allow_html=True)
                                        else:
                                            st.markdown('<p class="success-text">Second classification complete ✓</p>', unsafe_allow_html=True)
                                            
                                    except Exception as e:
                                        st.markdown(f'<p class="warning-text">Second classification failed: {str(e)}</p>', unsafe_allow_html=True)
                                        result2 = {"Items Style": "CLASSIFICATION_ERROR", "Class Broader Category": "CLASSIFICATION_ERROR"}
                            
                            # Verify and combine results
                            final_result = result1
                            confidence = "Low"
                            
                            if double_check and result2 and "ERROR" not in result2.get('Items Style', '') and "MISSING" not in result2.get('Items Style', ''):
                                final_result, is_high_confidence = verify_classifications(result1, result2)
                                confidence = "High" if is_high_confidence else "Medium"
                                
                                # Show verification results
                                st.markdown('<div class="double-check-result">', unsafe_allow_html=True)
                                st.markdown(f"**First Classification:** {result1['Items Style']} / {result1['Class Broader Category']}")
                                st.markdown(f"**Second Classification:** {result2['Items Style']} / {result2['Class Broader Category']}")
                                st.markdown(f"**Confidence:** {confidence}")
                                st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Update the dataframe with results
                            df.at[i, 'Items Style'] = final_result['Items Style']
                            df.at[i, 'Class Broader Category'] = final_result['Class Broader Category']
                            df.at[i, 'Classification Confidence'] = confidence
                            
                            # Add alternative classifications if present
                            if 'Alternative Style' in final_result:
                                df.at[i, 'Alternative Style'] = final_result['Alternative Style']
                            if 'Alternative Category' in final_result:
                                df.at[i, 'Alternative Category'] = final_result['Alternative Category']
                            
                            # Display the final result
                            st.markdown(f"**Final Style Classification:** {final_result['Items Style']}")
                            st.markdown(f"**Final Category Classification:** {final_result['Class Broader Category']}")
                            
                            # Show confidence indicator
                            if confidence == "High":
                                st.markdown('<p class="success-text">High Confidence Classification ✓✓</p>', unsafe_allow_html=True)
                            elif confidence == "Medium":
                                st.markdown('<p class="info-text">Medium Confidence Classification ✓</p>', unsafe_allow_html=True)
                            else:
                                st.markdown('<p class="warning-text">Low Confidence Classification ⚠️</p>', unsafe_allow_html=True)
                            
                            # Add a retry button for failed classifications or low confidence
                            if "ERROR" in final_result.get('Items Style', '') or "MISSING" in final_result.get('Items Style', '') or confidence == "Low":
                                retry_col = st.columns(1)[0]
                                with retry_col:
                                    if st.button(f"Retry item {i+1}"):
                                        # This will run on the next rerun
                                        pass
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Add a separator
                    st.markdown("---")
                    
                    # Update progress bar
                    progress_bar.progress((i - start_index + 1) / (len(df) - start_index))
                    
                    # Save dataframe periodically
                    if i % batch_size == 0 or i == len(df) - 1:
                        temp_result_path = os.path.join(tempfile.gettempdir(), "processed_results_temp.xlsx")
                        df.to_excel(temp_result_path, index=False)
                        st.info(f"Progress saved. Processed {i - start_index + 1} out of {len(df) - start_index} items.")
                
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
                
                # Summary statistics
                st.subheader("Classification Summary")
                
                # Count confidence levels
                confidence_counts = df['Classification Confidence'].value_counts()
                
                # Create a summary card
                st.markdown("""
                <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px;">
                    <h4>Classification Results</h4>
                """, unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Items Processed", len(df) - start_index)
                with col2:
                    high_conf = confidence_counts.get('High', 0)
                    st.metric("High Confidence Results", high_conf)
                with col3:
                    med_conf = confidence_counts.get('Medium', 0)
                    st.metric("Medium Confidence Results", med_conf)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
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
        "Class Broader Category": ["", ""],
        "Classification Confidence": ["", ""],
        "Alternative Style": ["", ""],
        "Alternative Category": ["", ""]
    })
    
    st.dataframe(example_df)
    
    # Create an expandable section with setup instructions
    with st.expander("API Key Setup Instructions"):
        st.markdown("""
        ### Setting Up Your API Key
        
        For the app to work, you need to provide an OpenAI API key with access to o4-mini.
        
        **Enter in the sidebar (temporary)**
        
        Enter your API key in the text field in the sidebar and click "Verify API Key". 
        This will only last for the current session.
        """)
        # **Option 2: Create a .env file (permanent)**
        
        # 1. Create a file named `.env` in the same directory as this script
        # 2. Add this line to the file:
        
        # ```
        # OPENAI_API_KEY=your_api_key_here
        # ```
        
        # 3. Save the file
        
        # This is more secure and avoids having to enter the key each time.
        # """)
