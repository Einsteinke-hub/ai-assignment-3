# app_pytorch.py - Streamlit deployment for MNIST classifier
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import io

# Set page config
st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="🔢",
    layout="wide"
)

# Define the CNN model (same as in Task 2)
class MNISTCNN(nn.Module):
    def __init__(self):
        super(MNISTCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout1(x)
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

@st.cache_resource
def load_model():
    """Load the pre-trained MNIST model"""
    model = MNISTCNN()
    
    # For demo purposes, we'll create a random model
    # In a real scenario, you would load a trained model:
    # model.load_state_dict(torch.load('mnist_model.pth', map_location='cpu'))
    
    model.eval()
    return model

def preprocess_image(image):
    """Preprocess uploaded image for the model"""
    # Convert to grayscale
    if image.mode != 'L':
        image = image.convert('L')
    
    # Resize to 28x28
    image = image.resize((28, 28))
    
    # Convert to numpy array and normalize
    image_array = np.array(image) / 255.0
    
    # Invert colors if background is dark
    if np.mean(image_array) < 0.5:
        image_array = 1 - image_array
    
    # Add batch and channel dimensions
    image_tensor = torch.FloatTensor(image_array).unsqueeze(0).unsqueeze(0)
    
    return image_tensor, image_array

def create_drawing_canvas():
    """Create a canvas for drawing digits"""
    st.subheader("Draw a Digit")
    col1, col2 = st.columns(2)
    
    with col1:
        # Canvas size
        canvas_size = 280
        image = Image.new('L', (canvas_size, canvas_size), color=255)
        draw = ImageDraw.Draw(image)
        
        st.markdown("**Draw in the area below:**")
        drawing = st.canvas(
            stroke_width=20,
            stroke_color="#000000",
            background_color="#FFFFFF",
            height=canvas_size,
            width=canvas_size,
            drawing_mode="freedraw",
            key="canvas"
        )
        
        if drawing is not None:
            # Convert the drawing to image
            if drawing.image_data is not None:
                image = Image.fromarray(drawing.image_data.astype('uint8'), 'RGBA')
                image = image.convert('L')
    
    with col2:
        st.markdown("**Preview & Prediction:**")
        if drawing is not None:
            # Display the drawn image
            st.image(image, width=150, caption="Your drawing")
            
            # Preprocess and predict
            image_tensor, processed_array = preprocess_image(image)
            
            return image_tensor, processed_array
    
    return None, None

def main():
    st.title("🎯 MNIST Handwritten Digit Classifier")
    st.markdown("""
    This app recognizes handwritten digits (0-9) using a Convolutional Neural Network.
    You can either upload an image or draw a digit directly in the app!
    """)
    
    # Load model
    model = load_model()
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["📁 Upload Image", "✏️ Draw Digit"])
    
    with tab1:
        st.header("Upload an Image")
        st.markdown("Upload an image of a handwritten digit (recommended: 28x28 pixels, white background)")
        
        uploaded_file = st.file_uploader(
            "Choose an image file", 
            type=["png", "jpg", "jpeg"],
            key="uploader"
        )
        
        if uploaded_file is not None:
            # Load and display image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width=200)
            
            # Preprocess and predict
            image_tensor, processed_array = preprocess_image(image)
            
            # Make prediction
            with torch.no_grad():
                output = model(image_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                predicted_class = torch.argmax(output[0]).item()
                confidence = probabilities[predicted_class].item()
            
            # Display results
            st.success(f"**Prediction: {predicted_class}**")
            st.info(f"**Confidence: {confidence:.2%}**")
            
            # Show probability distribution
            st.subheader("Probability Distribution")
            fig, ax = plt.subplots(figsize=(10, 4))
            digits = list(range(10))
            prob_values = probabilities.numpy()
            
            bars = ax.bar(digits, prob_values, color='skyblue', alpha=0.7)
            ax.set_xlabel('Digits')
            ax.set_ylabel('Probability')
            ax.set_title('Prediction Probabilities for Each Digit')
            ax.set_xticks(digits)
            
            # Highlight the predicted digit
            bars[predicted_class].set_color('red')
            
            # Add value labels on bars
            for i, v in enumerate(prob_values):
                ax.text(i, v + 0.01, f'{v:.2%}', ha='center', va='bottom')
            
            st.pyplot(fig)
    
    with tab2:
        st.header("Draw a Digit")
        st.markdown("Use your mouse or touchscreen to draw a digit in the canvas below")
        
        image_tensor, processed_array = create_drawing_canvas()
        
        if image_tensor is not None:
            # Make prediction
            with torch.no_grad():
                output = model(image_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                predicted_class = torch.argmax(output[0]).item()
                confidence = probabilities[predicted_class].item()
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.success(f"**Predicted Digit: {predicted_class}**")
                st.info(f"**Confidence: {confidence:.2%}**")
                
                if confidence > 0.7:
                    st.balloons()
            
            with col2:
                # Show processed image (28x28)
                st.image(processed_array, width=100, caption="Processed image (28x28)", clamp=True)
            
            # Show detailed probabilities
            st.subheader("Detailed Probabilities:")
            prob_cols = st.columns(5)
            for i in range(10):
                with prob_cols[i % 5]:
                    st.metric(
                        label=f"Digit {i}",
                        value=f"{probabilities[i]:.2%}",
                        delta="🏆" if i == predicted_class else None
                    )

    # Add sidebar with information
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This MNIST classifier uses a CNN architecture:
        
        - **Input**: 28×28 grayscale images
        - **Conv Layers**: 32 and 64 filters
        - **Fully Connected**: 128 neurons
        - **Output**: 10 classes (0-9)
        
        **Note**: This is a demo version with a randomly initialized model.
        For production, train on the actual MNIST dataset.
        """)
        
        st.header("How to Use")
        st.markdown("""
        1. **Upload**: Use clear images with white background
        2. **Draw**: Draw digits in the center of the canvas
        3. **Best Results**: Clear, centered digits work best
        """)
        
        if st.button("Clear Drawing", key="clear"):
            st.rerun()

if __name__ == "__main__":
    main()