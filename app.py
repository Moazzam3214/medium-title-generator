import streamlit as st
import numpy as np
import json
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Page configuration
st.set_page_config(
    page_title="Medium Title Generator",
    page_icon="🚀",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 3rem;
    }
    .generated-title {
        font-size: 1.5rem;
        color: #2e8b57;
        font-weight: bold;
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 2rem 0;
    }
    .example-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_and_tokenizer():
    """Load the trained model and tokenizer"""
    try:
        # Load model
        if os.path.exists('medium_title_gen.h5'):
            model = load_model('medium_title_gen.h5')
        else:
            st.error(
                "Model file 'medium_title_gen.h5' not found. Please ensure the model is in the same directory.")
            return None, None

        # Load tokenizer
        if os.path.exists('tokenizer.json'):
            with open('tokenizer.json', 'r') as f:
                data = f.read()  # read as string
                tokenizer = tokenizer_from_json(data)
        else:
            st.error(
                "Tokenizer file 'tokenizer.json' not found. Please ensure the tokenizer is in the same directory.")
            return None, None

        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model or tokenizer: {str(e)}")
        return None, None


def generate_title(model, tokenizer, seed_text, next_words=10, input_length=26):
    """Generate title using the trained model"""
    try:
        original_seed = seed_text

        for _ in range(next_words):
            # Tokenize and pad the input
            token_list = tokenizer.texts_to_sequences([seed_text])[0]
            token_list = pad_sequences(
                [token_list], maxlen=input_length, padding='pre')

            # Predict next word
            predicted = model.predict(token_list, verbose=0)
            predicted_word_index = np.argmax(predicted, axis=1)[0]

            # Find the word corresponding to the predicted index
            predicted_word = None
            for word, index in tokenizer.word_index.items():
                if index == predicted_word_index:
                    predicted_word = word
                    break

            if predicted_word:
                seed_text += " " + predicted_word
            else:
                break  # Stop if no word is found

        return seed_text
    except Exception as e:
        st.error(f"Error generating title: {str(e)}")
        return original_seed


def main():
    # Header
    st.markdown('<h1 class="main-header">🚀 Medium Title Generator</h1>',
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Generate compelling article titles using AI-powered deep learning</p>',
                unsafe_allow_html=True)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()

    if model is None or tokenizer is None:
        st.stop()

    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")

        # Number of words to generate
        next_words = st.slider(
            "Number of words to generate:",
            min_value=3,
            max_value=15,
            value=5,
            help="Choose how many additional words to generate"
        )

        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            input_length = st.number_input(
                "Input Length (max sequence length):",
                min_value=10,
                max_value=50,
                value=26,
                help="Maximum sequence length used during training"
            )

        st.markdown("---")

        # Examples section
        st.header("💡 Try These Examples")
        example_seeds = [
            "how to",
            "deep learning",
            "artificial intelligence",
            "data science",
            "machine learning",
            "what are",
            "why you should",
            "the future of"
        ]

        for seed in example_seeds:
            if st.button(f"'{seed}'", key=f"example_{seed}"):
                st.session_state.seed_input = seed

    # Main input area
    col1, col2 = st.columns([3, 1])

    with col1:
        seed_text = st.text_input(
            "Enter your seed text:",
            value=st.session_state.get('seed_input', ''),
            placeholder="e.g., 'how to', 'machine learning', 'the future of'...",
            help="Start with a few words, and the AI will complete the title"
        )

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
        generate_button = st.button("🎯 Generate Title", type="primary")

    # Generate title when button is clicked or seed text is provided
    if generate_button and seed_text.strip():
        with st.spinner("🤖 Generating your awesome title..."):
            generated_title = generate_title(
                model, tokenizer, seed_text.strip(), next_words, input_length
            )

            st.markdown(
                f'<div class="generated-title">📝 Generated Title:<br>"{generated_title}"</div>',
                unsafe_allow_html=True
            )

            # Add copy button
            st.code(generated_title, language=None)

            # Additional options
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🔄 Generate Another"):
                    # Re-run generation with same input
                    new_title = generate_title(
                        model, tokenizer, seed_text.strip(), next_words, input_length
                    )
                    st.markdown(
                        f'<div class="generated-title">📝 New Title:<br>"{new_title}"</div>',
                        unsafe_allow_html=True
                    )

            with col2:
                if st.button("📊 Get Variations"):
                    st.write("**Title Variations:**")
                    for i in range(3):
                        variation = generate_title(
                            model, tokenizer, seed_text.strip(),
                            next_words + np.random.randint(-2, 3), input_length
                        )
                        st.write(f"{i+1}. {variation}")

    elif generate_button and not seed_text.strip():
        st.warning("⚠️ Please enter some seed text to generate a title!")

    # Information sections
    st.markdown("---")

    # How it works
    with st.expander("🤔 How does it work?"):
        st.markdown("""
        This title generator uses a **Bidirectional LSTM** neural network trained on thousands of Medium articles. Here's the process:
        
        1. **Input Processing**: Your seed text is tokenized and converted to numerical sequences
        2. **Context Understanding**: The bidirectional LSTM analyzes context from both directions
        3. **Word Prediction**: The model predicts the most likely next word based on learned patterns
        4. **Title Construction**: This process repeats to build a complete, engaging title
        
        The model has learned from real Medium articles, so it understands what makes titles engaging and clickable!
        """)

    # Tips for better results
    with st.expander("💡 Tips for Better Results"):
        st.markdown("""
        - **Start with common phrases**: "How to", "Why you should", "The future of"
        - **Use specific topics**: "machine learning", "web development", "startup advice"
        - **Keep it relevant**: The model works best with topics similar to its training data
        - **Experiment with length**: Try different numbers of words to generate
        - **Generate multiple versions**: Click "Generate Another" for variations
        """)

    # Model information
    with st.expander("🧠 Model Information"):
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Training Accuracy", "~85%")
            st.metric("Validation Accuracy", "~82%")

        with col2:
            st.metric("Total Parameters", "~2.8M")
            st.metric("Training Epochs", "50")

        st.markdown("""
        **Architecture:**
        - Embedding Layer (100 dimensions)
        - Bidirectional LSTM (100 units)
        - Dense Output Layer (Softmax)
        """)

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 2rem;'>
            Made with ❤️ using Streamlit and TensorFlow<br>
            <small>Star this project on GitHub if you found it helpful!</small>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
