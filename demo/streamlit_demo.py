"""Streamlit demo for multi-modal domain adaptation."""

import streamlit as st
import torch
import numpy as np
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
import logging

from src.models.clip_domain_adaptation import CLIPDomainAdaptationModel
from src.eval.domain_adaptation_evaluator import DomainAdaptationEvaluator
from src.utils import get_device, load_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Multi-Modal Domain Adaptation Demo",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Safety disclaimer
SAFETY_DISCLAIMER = """
⚠️ **Important Disclaimer**: This model is for research and educational purposes only. 
Not intended for medical diagnosis or clinical use. Results should be validated by qualified professionals.
"""

def load_model(checkpoint_path: str, config_path: str):
    """Load the domain adaptation model."""
    try:
        # Load configuration
        config = load_config(config_path)
        
        # Create model
        model = CLIPDomainAdaptationModel(
            base_model_name=config.model.base_model_name,
            config=config.model
        )
        
        # Load checkpoint
        device = get_device("auto")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        
        return model, device, config
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

def encode_image_text(model, image, text, device):
    """Encode image and text using the model."""
    try:
        with torch.no_grad():
            # Process inputs
            inputs = model.processor(
                text=[text],
                images=[image],
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            # Move to device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
            
            # Forward pass
            outputs = model(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                attention_mask=inputs["attention_mask"],
                return_embeddings=True
            )
            
            return outputs["image_embeds"], outputs["text_embeds"]
    except Exception as e:
        st.error(f"Error encoding inputs: {str(e)}")
        return None, None

def main():
    """Main Streamlit app."""
    
    # Header
    st.markdown('<h1 class="main-header">🔬 Multi-Modal Domain Adaptation Demo</h1>', 
                unsafe_allow_html=True)
    
    # Safety disclaimer
    st.markdown(f'<div class="warning-box">{SAFETY_DISCLAIMER}</div>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # Model selection
    checkpoint_path = st.sidebar.text_input(
        "Model Checkpoint Path",
        value="outputs/checkpoints/best_model.pt",
        help="Path to the trained model checkpoint"
    )
    
    config_path = st.sidebar.text_input(
        "Config Path",
        value="configs/config.yaml",
        help="Path to the configuration file"
    )
    
    # Load model button
    if st.sidebar.button("Load Model"):
        with st.spinner("Loading model..."):
            model, device, config = load_model(checkpoint_path, config_path)
            if model is not None:
                st.sidebar.success("Model loaded successfully!")
                st.session_state.model = model
                st.session_state.device = device
                st.session_state.config = config
            else:
                st.sidebar.error("Failed to load model")
    
    # Check if model is loaded
    if "model" not in st.session_state:
        st.warning("Please load a model first using the sidebar.")
        return
    
    model = st.session_state.model
    device = st.session_state.device
    config = st.session_state.config
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Image-Text Matching", 
        "📊 Domain Analysis", 
        "📈 Model Performance", 
        "ℹ️ About"
    ])
    
    with tab1:
        st.header("Image-Text Matching")
        st.write("Upload an image and enter text to see how well they match across domains.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Input")
            
            # Image upload
            uploaded_image = st.file_uploader(
                "Upload an image",
                type=['png', 'jpg', 'jpeg'],
                help="Upload an image to analyze"
            )
            
            if uploaded_image is not None:
                image = Image.open(uploaded_image).convert("RGB")
                st.image(image, caption="Uploaded Image", use_column_width=True)
            else:
                # Default image
                image = Image.new("RGB", (224, 224), color=(128, 128, 128))
                st.image(image, caption="Default Image", use_column_width=True)
            
            # Text input
            text_input = st.text_area(
                "Enter text description",
                value="A medical image showing healthy tissue",
                help="Enter a text description to match with the image"
            )
            
            # Domain selection
            domain = st.selectbox(
                "Domain",
                ["general", "medical"],
                help="Select the domain context"
            )
        
        with col2:
            st.subheader("Analysis")
            
            if st.button("Analyze Match"):
                with st.spinner("Analyzing..."):
                    # Encode image and text
                    image_embeds, text_embeds = encode_image_text(model, image, text_input, device)
                    
                    if image_embeds is not None and text_embeds is not None:
                        # Compute similarity
                        similarity = torch.cosine_similarity(image_embeds, text_embeds).item()
                        
                        # Display results
                        st.metric("Similarity Score", f"{similarity:.3f}")
                        
                        # Similarity bar chart
                        fig = go.Figure(go.Bar(
                            x=["Similarity"],
                            y=[similarity],
                            marker_color=['green' if similarity > 0.5 else 'orange' if similarity > 0.3 else 'red']
                        ))
                        fig.update_layout(
                            title="Image-Text Similarity",
                            yaxis_title="Similarity Score",
                            yaxis=dict(range=[0, 1])
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Interpretation
                        if similarity > 0.7:
                            st.success("Strong match! The image and text are well-aligned.")
                        elif similarity > 0.4:
                            st.warning("Moderate match. Some alignment between image and text.")
                        else:
                            st.error("Weak match. Poor alignment between image and text.")
    
    with tab2:
        st.header("Domain Analysis")
        st.write("Analyze how the model handles different domains and domain adaptation.")
        
        # Domain comparison
        st.subheader("Domain Embedding Analysis")
        
        # Sample texts for each domain
        general_texts = [
            "A photo of a dog",
            "A picture of a car",
            "An image of a house",
            "A photo of food"
        ]
        
        medical_texts = [
            "A medical scan showing healthy tissue",
            "An X-ray displaying normal anatomy",
            "A CT scan revealing pathology",
            "An MRI scan of the brain"
        ]
        
        if st.button("Analyze Domain Embeddings"):
            with st.spinner("Computing domain embeddings..."):
                # Create dummy image for text-only analysis
                dummy_image = Image.new("RGB", (224, 224), color=(128, 128, 128))
                
                general_embeddings = []
                medical_embeddings = []
                
                # Encode general domain texts
                for text in general_texts:
                    _, text_embeds = encode_image_text(model, dummy_image, text, device)
                    if text_embeds is not None:
                        general_embeddings.append(text_embeds.cpu().numpy())
                
                # Encode medical domain texts
                for text in medical_texts:
                    _, text_embeds = encode_image_text(model, dummy_image, text, device)
                    if text_embeds is not None:
                        medical_embeddings.append(text_embeds.cpu().numpy())
                
                if general_embeddings and medical_embeddings:
                    # Compute average embeddings
                    general_avg = np.mean(general_embeddings, axis=0)
                    medical_avg = np.mean(medical_embeddings, axis=0)
                    
                    # Compute domain separation
                    domain_similarity = np.dot(general_avg.flatten(), medical_avg.flatten())
                    
                    st.metric("Domain Separation Score", f"{1 - domain_similarity:.3f}")
                    
                    # Domain similarity matrix
                    domain_matrix = np.array([
                        [1.0, domain_similarity],
                        [domain_similarity, 1.0]
                    ])
                    
                    fig = px.imshow(
                        domain_matrix,
                        labels=dict(x="Domain", y="Domain", color="Similarity"),
                        x=["General", "Medical"],
                        y=["General", "Medical"],
                        color_continuous_scale="RdBu_r",
                        title="Domain Similarity Matrix"
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Model Performance")
        st.write("View model performance metrics and evaluation results.")
        
        # Performance metrics
        st.subheader("Performance Metrics")
        
        # Sample metrics (in a real app, these would be loaded from evaluation results)
        metrics = {
            "Retrieval Performance": {
                "Recall@1": 0.756,
                "Recall@5": 0.892,
                "Recall@10": 0.934,
                "Median Rank": 2.3
            },
            "Domain Adaptation": {
                "Domain Alignment Score": 0.823,
                "Domain Separation Score": 0.678,
                "Transfer Accuracy": 0.845
            },
            "Contrastive Learning": {
                "Alignment Loss": 0.234,
                "Uniformity Loss": 0.456
            }
        }
        
        # Display metrics in columns
        for category, category_metrics in metrics.items():
            st.subheader(category)
            
            cols = st.columns(len(category_metrics))
            for i, (metric, value) in enumerate(category_metrics.items()):
                with cols[i]:
                    st.metric(metric, f"{value:.3f}")
        
        # Performance visualization
        st.subheader("Performance Trends")
        
        # Sample training history (in a real app, this would be loaded from training logs)
        epochs = list(range(1, 51))
        train_losses = [0.8 - 0.6 * np.exp(-epoch/20) + 0.1 * np.random.random() for epoch in epochs]
        val_losses = [0.7 - 0.5 * np.exp(-epoch/25) + 0.15 * np.random.random() for epoch in epochs]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=epochs, y=train_losses, mode='lines', name='Training Loss'))
        fig.add_trace(go.Scatter(x=epochs, y=val_losses, mode='lines', name='Validation Loss'))
        
        fig.update_layout(
            title="Training Progress",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("About This Demo")
        
        st.markdown("""
        ## Multi-Modal Domain Adaptation
        
        This demo showcases a multi-modal domain adaptation model that learns to transfer 
        knowledge from a general domain (everyday images and text) to a specialized domain 
        (medical images and descriptions).
        
        ### Key Features:
        
        - **Domain Adaptation**: Adapts pre-trained CLIP models to new domains using adapter layers
        - **Contrastive Learning**: Uses InfoNCE loss for learning aligned representations
        - **Cross-Modal Retrieval**: Enables image-to-text and text-to-image retrieval
        - **Parameter Efficiency**: Uses adapter layers for efficient fine-tuning
        
        ### Model Architecture:
        
        - **Base Model**: CLIP (Vision-Language model)
        - **Adaptation Method**: Adapter layers + domain-specific projections
        - **Loss Function**: Contrastive loss + domain alignment + uniformity loss
        
        ### Use Cases:
        
        - Medical image analysis and description
        - Cross-domain image-text retrieval
        - Domain-specific visual question answering
        - Transfer learning for specialized domains
        
        ### Technical Details:
        
        - **Framework**: PyTorch 2.x
        - **Model Size**: ~150M parameters
        - **Training**: Contrastive learning with domain adaptation
        - **Evaluation**: Retrieval metrics + domain alignment scores
        
        ### Safety & Limitations:
        
        - This model is for research and educational purposes only
        - Not intended for medical diagnosis or clinical use
        - Results should be validated by qualified professionals
        - Model may have biases inherited from training data
        
        ### Repository:
        
        For more details, visit: [github.com/kryptologyst](https://github.com/kryptologyst)
        """)

if __name__ == "__main__":
    main()
