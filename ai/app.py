"""
Streamlit App for LeNet-5 Testing
Supports 2 modes:
1. Realtime Drawing - Draw digits/shapes and predict
2. Upload File - Upload image and predict
"""

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import cv2
import sys
import os
import pickle

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.utils.LayerObjects import LeNet5
from ai.utils.utils_func import zero_pad, normalize

# Page config
st.set_page_config(
    page_title="LeNet-5 Digit & Shape Classifier",
    page_icon="🔢",
    layout="wide"
)


# Class names mapping (default: digits + shapes)
DEFAULT_CLASS_NAMES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'Circle', 'Square', 'Triangle']

# Allow user to input custom class names (for HASYv2 or other models)
st.sidebar.markdown("---")
st.sidebar.markdown("**Custom CLASS_NAMES (comma-separated):**")
custom_class_names = st.sidebar.text_input(
    "CLASS_NAMES:",
    value=','.join(DEFAULT_CLASS_NAMES),
    help="Nhập nhãn theo đúng thứ tự model đã train. VD: 0,1,2,3,4,5,6,7,8,9,+,-,*,/,(,)"
)
CLASS_NAMES = [x.strip() for x in custom_class_names.split(',') if x.strip()]

@st.cache_resource
def load_model(model_path):
    """Load trained LeNet-5 model"""
    try:
        with open(model_path, 'rb') as f:
            model_state = pickle.load(f)
        
        # Initialize model
        C3_mapping = [[0,1,2],[1,2,3],[2,3,4],[3,4,5],[4,5,0],[5,0,1],
                      [0,1,2,3],[1,2,3,4],[2,3,4,5],[3,4,5,0],[4,5,0,1],[5,0,1,2],
                      [0,1,3,4],[1,2,4,5],[0,2,3,5],
                      [0,1,2,3,4,5]]
        
        n_classes = model_state.get('n_classes', 13)
        model = LeNet5(n_classes=n_classes, C3_mapping=C3_mapping)
        
        # Restore weights
        model.C1.weight = model_state['C1_weight']
        model.C1.bias = model_state['C1_bias']
        model.C3.wb = model_state['C3_wb']
        model.C5.weight = model_state['C5_weight']
        model.C5.bias = model_state['C5_bias']
        model.F6.weight = model_state['F6_weight']
        model.F6.bias = model_state['F6_bias']
        model.RBF.weight = model_state['RBF_weight']
        
        return model, n_classes
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def preprocess_image(image, apply_threshold=False):
    """Preprocess image for LeNet-5 input"""
    # Convert to grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply threshold to make lines clearer (optional)
    if apply_threshold:
        _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    
    # Resize to 28x28 with anti-aliasing
    image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Apply preprocessing
    img_input = normalize(zero_pad(image[np.newaxis, :, :, np.newaxis], 2), 'lenet5')
    
    return img_input

def resize_drawing_smart(image, target_size=28, source_size=280):
    """Smart resize for hand-drawn images with proper scaling"""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # Step 1: Find bounding box of drawing to center it
    coords = cv2.findNonZero(gray)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        
        # Add padding (10% of image size)
        padding = int(source_size * 0.1)
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(source_size - x, w + 2 * padding)
        h = min(source_size - y, h + 2 * padding)
        
        # Crop to bounding box
        cropped = gray[y:y+h, x:x+w]
        
        # Make it square (maintain aspect ratio)
        size = max(w, h)
        square = np.zeros((size, size), dtype=np.uint8)
        offset_x = (size - w) // 2
        offset_y = (size - h) // 2
        square[offset_y:offset_y+h, offset_x:offset_x+w] = cropped
        
        gray = square
    
    # Step 2: Apply light Gaussian blur to smooth edges
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Step 3: Resize with high-quality interpolation
    # Use INTER_AREA for downsampling (best quality)
    resized = cv2.resize(blurred, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
    
    # Step 4: Enhance contrast with CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(resized)
    
    # Step 5: Apply adaptive thresholding to make lines clearer
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return binary

def predict_image(model, image, use_smart_resize=False):
    """Predict class for given image"""
    if use_smart_resize:
        # Use smart resize for hand-drawn images
        resized = resize_drawing_smart(image, target_size=28)
        img_input = normalize(zero_pad(resized[np.newaxis, :, :, np.newaxis], 2), 'lenet5')
    else:
        img_input = preprocess_image(image, apply_threshold=False)
    
    _, prediction = model.Forward_Propagation(img_input, np.array([0]), mode='test')
    return prediction[0]

def get_confidence_scores(model, image, use_smart_resize=False):
    """Get confidence scores for all classes"""
    if use_smart_resize:
        resized = resize_drawing_smart(image, target_size=28)
        img_input = normalize(zero_pad(resized[np.newaxis, :, :, np.newaxis], 2), 'lenet5')
    else:
        img_input = preprocess_image(image, apply_threshold=False)
    
    # Forward propagation to get F6 output
    model.Forward_Propagation(img_input, np.array([0]), mode='test')
    
    # Get F6 output (after activation) - this is input to RBF layer
    f6_output = model.a4_FP[0]  # Shape: (84,)
    
    # Calculate RBF distances for all classes
    # Distance = ||F6_output - RBF_weight||^2
    distances = np.sum((f6_output - model.RBF.weight) ** 2, axis=1)  # Shape: (n_classes,)
    
    # Convert distances to pseudo-probabilities
    # Lower distance = higher probability
    # Use softmax-like transformation
    scores = np.exp(-distances / distances.std())  # Exponential of negative distances
    scores = scores / (scores.sum() + 1e-10)  # Normalize to sum=1
    
    return scores

def segment_characters(image):
    """Segment individual characters from an image with improved preprocessing"""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # Step 1: Detect if image is inverted (white text on black background)
    mean_val = np.mean(gray)
    is_inverted = mean_val < 127
    if is_inverted:  # Dark background, invert it to white background
        gray = cv2.bitwise_not(gray)
    
    # Step 2: Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Step 3: Apply adaptive threshold
    # Since we already inverted to white background, use THRESH_BINARY_INV to get white digits on black
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Step 4: Morphological operations to clean up
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Step 5: Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Step 6: Get bounding boxes and filter
    bounding_boxes = []
    img_area = gray.shape[0] * gray.shape[1]
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        contour_area = cv2.contourArea(contour)
        
        # Filter out noise and too large objects
        # Size should be reasonable (not too small, not too large)
        if w > 10 and h > 10 and area < img_area * 0.5 and contour_area > 50:
            # Check aspect ratio (should be reasonable for digits)
            aspect_ratio = w / float(h)
            if 0.15 < aspect_ratio < 2.0:  # Reasonable for digits
                bounding_boxes.append((x, y, w, h))
    
    # Step 7: Sort by x coordinate (left to right)
    bounding_boxes = sorted(bounding_boxes, key=lambda box: box[0])
    
    # Step 8: Extract and preprocess each character
    characters = []
    for x, y, w, h in bounding_boxes:
        # Add dynamic padding (10-20% of size)
        padding = max(8, int(max(w, h) * 0.2))
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(gray.shape[1], x + w + padding)
        y2 = min(gray.shape[0], y + h + padding)
        
        # Extract character from original grayscale (already inverted if needed)
        char_img = gray[y1:y2, x1:x2]
        
        # Invert to black digits on white background (MNIST style)
        char_img = cv2.bitwise_not(char_img)
        
        # Make square with proper centering
        size = max(char_img.shape)
        square = np.zeros((size, size), dtype=np.uint8)  # Black background
        offset_x = (size - char_img.shape[1]) // 2
        offset_y = (size - char_img.shape[0]) // 2
        square[offset_y:offset_y+char_img.shape[0], offset_x:offset_x+char_img.shape[1]] = char_img
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(square)
        
        # Resize to 28x28 with best interpolation
        # INTER_AREA is best for downsampling
        # Add slight blur before resize to avoid aliasing
        smooth = cv2.GaussianBlur(enhanced, (3, 3), 0)
        resized = cv2.resize(smooth, (28, 28), interpolation=cv2.INTER_AREA)
        
        # Normalize like MNIST (black background, white digits)
        # Apply threshold to clean up
        _, clean = cv2.threshold(resized, 30, 255, cv2.THRESH_BINARY)
        
        characters.append((clean, (x, y, w, h)))
    
    return characters

def recognize_expression(model, image):
    """Recognize mathematical expression from image"""
    # Segment characters
    characters = segment_characters(image)
    
    if not characters:
        return None, None, []
    
    # Map of operators (if model supports them)
    # For now, we'll use digits only and ask user to specify operators
    operator_map = {
        10: '+',
        11: '-',
        12: '*',
        13: '/',
        14: '(',
        15: ')'
    }
    
    expression = ""
    char_predictions = []
    
    for char_img, bbox in characters:
        # Predict character
        prediction = predict_image(model, char_img, use_smart_resize=False)
        
        # Convert to character
        if prediction < 10:
            char = str(prediction)
        elif prediction in operator_map:
            char = operator_map[prediction]
        else:
            char = '?'
        
        expression += char
        char_predictions.append((char, bbox, char_img))
    
    # Calculate result
    try:
        # Use eval safely for math expressions
        result = eval(expression)
    except:
        result = None
    
    return expression, result, char_predictions

def safe_eval_expression(expression):
    """Safely evaluate mathematical expression"""
    try:
        # Remove any spaces
        expression = expression.replace(' ', '')
        
        # Check if expression contains only valid characters
        valid_chars = set('0123456789+-*/(). ')
        if not all(c in valid_chars for c in expression):
            return None, "Invalid characters in expression"
        
        # Evaluate
        result = eval(expression)
        return result, None
    except ZeroDivisionError:
        return None, "Division by zero"
    except SyntaxError:
        return None, "Invalid syntax"
    except Exception as e:
        return None, f"Error: {str(e)}"

# Sidebar - Model selection
st.sidebar.title("⚙️ Settings")
model_path = st.sidebar.text_input(
    "Model Path:",
    value=r"E:\PTIT\XLA\ai\models\model_weights_17.pkl",
    help="Path to trained model checkpoint"
)

# Load model
if os.path.exists(model_path):
    model, n_classes = load_model(model_path)
    if model:
        st.sidebar.success(f"✅ Model loaded ({n_classes} classes)")
    else:
        st.sidebar.error("❌ Failed to load model")
        model = None
else:
    st.sidebar.warning("⚠️ Model file not found")
    model = None

# Main title
st.title("🔢 LeNet-5 Classifier")
st.markdown("Test LeNet-5 model với 3 chế độ: **Realtime Drawing**, **Detect Numbers** hoặc **Upload File**")

# Mode selection
mode = st.radio(
    "Select Mode:",
    ["🎨 Realtime Drawing", "🔍 Detect Numbers", "📁 Upload File"],
    horizontal=True
)

# ==================== MODE 1: REALTIME DRAWING ====================
if mode == "🎨 Realtime Drawing":
    st.header("Draw a digit or shape")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Canvas")
        
        # Drawing canvas using streamlit-drawable-canvas
        try:
            from streamlit_drawable_canvas import st_canvas
            
            # Canvas configuration
            ACTUAL_SIZE = 28  # Target size
            DISPLAY_SIZE = 280  # Drawing size (10x larger)
            
            stroke_width = st.slider("Brush Size:", 5, 20, 10)
            stroke_color = "#FFFFFF"  # White
            bg_color = "#000000"  # Black
            
            # Preprocessing options
            col_a, col_b = st.columns(2)
            with col_a:
                use_smart_resize = st.checkbox("✨ Smart Resize", value=True,
                                              help="Auto center + enhance quality")
            with col_b:
                show_steps = st.checkbox("👁️ Show Steps", value=False,
                                        help="Show preprocessing steps")
            
            # Create a canvas component
            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 0.0)",
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                background_color=bg_color,
                height=DISPLAY_SIZE,
                width=DISPLAY_SIZE,
                drawing_mode="freedraw",
                key="canvas",
            )
            
            st.markdown("**Instructions:**")
            st.markdown("- Draw a digit (0-9) or shape (circle, square, triangle)")
            st.markdown("- Click **Predict** to classify")
            st.markdown("- Click **Clear** (🗑️) in canvas toolbar to reset")
            
        except ImportError:
            st.warning("⚠️ streamlit-drawable-canvas not installed")
            st.code("pip install streamlit-drawable-canvas", language="bash")
            canvas_result = None
    
    with col2:
        st.markdown("### Prediction")
        
        if model and canvas_result is not None and canvas_result.image_data is not None:
            # Get canvas image (DISPLAY_SIZE x DISPLAY_SIZE)
            img = canvas_result.image_data
            
            # Check if canvas is not empty
            if img[:, :, :3].sum() > 0:
                # Convert RGBA to RGB
                img_rgb = img[:, :, :3]
                
                # Apply smart resize
                if use_smart_resize:
                    img_gray = resize_drawing_smart(img_rgb, target_size=ACTUAL_SIZE, source_size=DISPLAY_SIZE)
                    caption = "Smart Resize (centered + enhanced)"
                else:
                    img_resized = cv2.resize(img_rgb, (ACTUAL_SIZE, ACTUAL_SIZE), 
                                            interpolation=cv2.INTER_AREA)
                    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
                    caption = "Basic Resize"
                
                # Show preprocessing steps if enabled
                if show_steps and use_smart_resize:
                    st.markdown("**Preprocessing Steps:**")
                    gray_temp = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
                    
                    col_step1, col_step2, col_step3 = st.columns(3)
                    with col_step1:
                        st.image(gray_temp, caption="1. Original 280×280", width=100)
                    with col_step2:
                        temp_resized = cv2.resize(gray_temp, (28, 28), interpolation=cv2.INTER_AREA)
                        st.image(temp_resized, caption="2. After Resize", width=100)
                    with col_step3:
                        st.image(img_gray, caption="3. Final (centered)", width=100)
                
                st.image(img_gray, caption=caption, width=150)
                
                # Predict button
                if st.button("🔮 Predict", type="primary", use_container_width=True):
                    with st.spinner("Predicting..."):
                        prediction = predict_image(model, img_rgb, use_smart_resize=use_smart_resize)
                        scores = get_confidence_scores(model, img_rgb, use_smart_resize=use_smart_resize)
                        
                        # Display prediction
                        st.success(f"### Prediction: **{CLASS_NAMES[prediction]}**")
                        
                        # Display confidence scores
                        st.markdown("#### Confidence Scores:")
                        for i, score in enumerate(scores):
                            st.progress(float(score), text=f"{CLASS_NAMES[i]}: {score*100:.1f}%")
            else:
                st.info("👆 Draw something on the canvas")
        elif not model:
            st.error("Model not loaded")

# ==================== MODE 2: DETECT NUMBERS ====================
elif mode == "🔍 Detect Numbers":
    st.header("Detect multiple numbers in image")
    st.info("💡 **Vẽ hoặc upload ảnh chứa nhiều số (0-9), app sẽ tự động phát hiện và nhận diện tất cả các số**")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Input")
        
        # Choose input method
        input_method = st.radio(
            "Choose input method:",
            ["✏️ Draw", "📤 Upload Image"],
            horizontal=True,
            key="detect_input_method"
        )
        
        # Preprocessing options
        show_preprocessing = st.checkbox("🔬 Show Preprocessing Steps", value=False, 
                                        help="Hiển thị các bước xử lý ảnh")
        
        if input_method == "✏️ Draw":
            try:
                from streamlit_drawable_canvas import st_canvas
                
                # Canvas configuration
                CALC_DISPLAY_SIZE = 600
                
                stroke_width = st.slider("Brush Size:", 5, 30, 15)
                stroke_color = "#FFFFFF"  # White
                bg_color = "#000000"  # Black
                
                # Create a canvas component
                calc_canvas_result = st_canvas(
                    fill_color="rgba(255, 255, 255, 0.0)",
                    stroke_width=stroke_width,
                    stroke_color=stroke_color,
                    background_color=bg_color,
                    height=CALC_DISPLAY_SIZE,
                    width=CALC_DISPLAY_SIZE,
                    drawing_mode="freedraw",
                    key="detect_canvas",
                )
                
                st.markdown("**Instructions:**")
                st.markdown("- Viết các số cách nhau một khoảng")
                st.markdown("- Click **Detect** để nhận diện")
                st.markdown("- Vẽ rõ ràng để nhận diện chính xác hơn")
                
            except ImportError:
                st.warning("⚠️ streamlit-drawable-canvas not installed")
                calc_canvas_result = None
        else:
            # Upload image
            detect_uploaded_file = st.file_uploader(
                "Upload image with multiple numbers",
                type=["png", "jpg", "jpeg", "bmp"],
                key="detect_upload"
            )
            
            if detect_uploaded_file is not None:
                # Load image
                detect_image = Image.open(detect_uploaded_file).convert('RGB')
                detect_img_array = np.array(detect_image)
                
                st.image(detect_image, caption=f"Uploaded Image ({detect_image.size[0]}×{detect_image.size[1]})", width=400)
            else:
                detect_img_array = None
                st.info("👆 Upload an image with multiple numbers")
    
    with col2:
        st.markdown("### Detected Numbers")
        
        # Process based on input method
        should_process = False
        img_to_process = None
        
        if input_method == "✏️ Draw":
            if model and calc_canvas_result is not None and calc_canvas_result.image_data is not None:
                img = calc_canvas_result.image_data
                if img[:, :, :3].sum() > 0:
                    img_to_process = img[:, :, :3]
                    should_process = st.button("🔍 Detect Numbers", type="primary", use_container_width=True)
                else:
                    st.info("👆 Draw numbers on the canvas")
            elif not model:
                st.error("Model not loaded")
        else:
            if detect_uploaded_file is not None and model:
                img_to_process = detect_img_array
                should_process = True
            elif not model:
                st.error("Model not loaded")
        
        # Process image
        if should_process and img_to_process is not None:
            with st.spinner("Detecting numbers..."):
                characters = segment_characters(img_to_process)
                
                if characters:
                    st.success(f"✅ Found **{len(characters)}** numbers")
                    
                    # Show preprocessing steps if enabled
                    if show_preprocessing and len(characters) > 0:
                        st.markdown("---")
                        st.markdown("#### 🔬 Preprocessing Example (First Character):")
                        
                        # Show preprocessing steps for first character
                        first_char, first_bbox = characters[0]
                        
                        # Get original crop
                        gray = cv2.cvtColor(img_to_process, cv2.COLOR_RGB2GRAY) if len(img_to_process.shape) == 3 else img_to_process
                        x, y, w, h = first_bbox
                        padding = max(5, int(max(w, h) * 0.15))
                        x1 = max(0, x - padding)
                        y1 = max(0, y - padding)
                        x2 = min(gray.shape[1], x + w + padding)
                        y2 = min(gray.shape[0], y + h + padding)
                        original_crop = gray[y1:y2, x1:x2]
                        
                        col_step1, col_step2, col_step3 = st.columns(3)
                        with col_step1:
                            st.image(original_crop, caption="1. Extracted Region", width=100)
                        with col_step2:
                            # Make square version
                            size = max(original_crop.shape)
                            square = np.ones((size, size), dtype=np.uint8) * 255
                            offset_x = (size - original_crop.shape[1]) // 2
                            offset_y = (size - original_crop.shape[0]) // 2
                            square[offset_y:offset_y+original_crop.shape[0], offset_x:offset_x+original_crop.shape[1]] = original_crop
                            st.image(square, caption="2. Centered Square", width=100)
                        with col_step3:
                            st.image(first_char, caption="3. Final 28×28", width=100)
                    
                    # Predict all characters
                    recognized_numbers = []
                    confidence_data = []
                    all_predictions = []  # Store all predictions for debugging
                    
                    for char_img, bbox in characters:
                        prediction = predict_image(model, char_img, use_smart_resize=False)
                        scores = get_confidence_scores(model, char_img, use_smart_resize=False)
                        all_predictions.append(prediction)
                        
                        # Accept numbers 0-9 and shapes if they're in valid range
                        if prediction < len(CLASS_NAMES):
                            predicted_class = CLASS_NAMES[prediction]
                            # Only add if it's a digit (not shape)
                            if prediction < 10:
                                recognized_numbers.append(str(prediction))
                                confidence_data.append((str(prediction), scores[prediction]))
                    
                    # Debug: show what was detected
                    if len(recognized_numbers) == 0:
                        st.warning(f"⚠️ Detected {len(characters)} objects but none recognized as digits (0-9)")
                        st.info(f"Predictions were: {[CLASS_NAMES[p] if p < len(CLASS_NAMES) else f'Unknown({p})' for p in all_predictions]}")
                    
                    # Display detected numbers
                    if len(recognized_numbers) > 0:
                        st.markdown("---")
                        st.markdown("### 🔢 Detected Numbers:")
                        
                        # Show as large text
                        numbers_str = ', '.join(recognized_numbers)
                        st.markdown(f"<h2 style='text-align: center; color: #4CAF50;'>{numbers_str}</h2>", 
                                   unsafe_allow_html=True)
                    
                    # Show individual detections with images
                    st.markdown("---")
                    st.markdown("#### Individual Detections:")
                    
                    # Display in grid
                    num_cols = min(5, len(characters))
                    cols = st.columns(num_cols)
                    
                    for idx, (char_img, bbox) in enumerate(characters):
                        with cols[idx % num_cols]:
                            prediction = all_predictions[idx]
                            predicted_class = CLASS_NAMES[prediction] if prediction < len(CLASS_NAMES) else f"Unknown({prediction})"
                            st.image(char_img, caption=f"{predicted_class}", width=80)
                    
                    # Show confidence scores
                    st.markdown("---")
                    st.markdown("#### Confidence Scores:")
                    if len(confidence_data) > 0:
                        for num, conf in confidence_data:
                            st.progress(float(conf), text=f"**{num}**: {conf*100:.1f}%")
                    else:
                        st.info("No digit confidence scores to display")
                    
                    # Summary
                    st.markdown("---")
                    st.markdown("#### 📊 Summary:")
                    st.markdown(f"- **Total objects detected**: {len(characters)}")
                    st.markdown(f"- **Numbers recognized**: {len(recognized_numbers)}")
                    if len(recognized_numbers) > 0:
                        numbers_str = ', '.join(recognized_numbers)
                        st.markdown(f"- **Numbers found**: {numbers_str}")
                    
                else:
                    st.warning("⚠️ No numbers detected. Try drawing more clearly or adjusting the image.")

# ==================== MODE 3: UPLOAD FILE ====================
elif mode == "📁 Upload File":
    st.header("Upload an image")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["png", "jpg", "jpeg", "bmp"],
            help="Upload a 28×28 grayscale image or any size (will be resized)"
        )
        
        if uploaded_file is not None:
            # Load image
            image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
            img_array = np.array(image)
            
            # Display original image
            st.image(image, caption=f"Original Image ({image.size[0]}×{image.size[1]})", width=300)
            
            # Display preprocessed image
            img_resized = cv2.resize(img_array, (28, 28), interpolation=cv2.INTER_AREA)
            st.image(img_resized, caption="Preprocessed (28×28)", width=150)
    
    with col2:
        st.markdown("### Prediction")
        
        if uploaded_file is not None and model:
            # Auto predict
            with st.spinner("Predicting..."):
                prediction = predict_image(model, img_array)
                scores = get_confidence_scores(model, img_array)
                
                # Display prediction
                st.success(f"### Prediction: **{CLASS_NAMES[prediction]}**")
                
                # Display confidence scores
                st.markdown("#### Confidence Scores:")
                for i, score in enumerate(scores):
                    st.progress(float(score), text=f"{CLASS_NAMES[i]}: {score*100:.1f}%")
                
                # Additional info
                st.markdown("---")
                st.markdown("#### Image Info:")
                st.markdown(f"- **Original Size**: {image.size[0]}×{image.size[1]}")
                st.markdown(f"- **Preprocessed**: 28×28")
                st.markdown(f"- **Predicted Class**: {CLASS_NAMES[prediction]} (label {prediction})")
        
        elif not model:
            st.error("Model not loaded")
        else:
            st.info("👆 Upload an image file")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><b>LeNet-5 from Scratch</b> - NumPy Implementation</p>
    <p>✨ Supports: Single digit recognition + Multiple numbers detection</p>
    <p>📊 Classes: 0-9 digits (MNIST) + Circle, Square, Triangle</p>
</div>
""", unsafe_allow_html=True)

# Sidebar - Additional info
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📖 How to Use")
    st.markdown("""
    **Realtime Drawing:**
    1. Draw với chuột/touchpad
    2. Adjust brush size nếu cần
    3. Click **Predict**
    4. Click **Clear** để vẽ lại
    
    **Detect Numbers:**
    1. Vẽ hoặc upload ảnh có nhiều số
    2. Click **Detect** để nhận diện
    3. Xem kết quả các số được phát hiện
    
    **Upload File:**
    1. Click **Browse files**
    2. Choose image (PNG, JPG)
    3. Auto predict
    
    **Tips:**
    - Vẽ rõ ràng, kích thước vừa phải
    - Các số cách nhau để dễ nhận diện
    - Shapes: Circle = tròn, Square = vuông, Triangle = tam giác
    - Digits: 0-9 như MNIST
    """)
    
    st.markdown("---")
    st.markdown("### 🔧 Model Info")
    if model:
        st.markdown(f"- **Classes**: {n_classes}")
        st.markdown(f"- **Architecture**: LeNet-5")
        st.markdown(f"- **Input**: 28×28 grayscale")
        st.markdown(f"- **Output**: {', '.join(CLASS_NAMES[:n_classes])}")