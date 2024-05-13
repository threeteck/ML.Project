import streamlit as st
import pickle
import torchtext
from run_model import build_model, run_model, DataProcessor

# Load vocabularies and build the processor
def load_resources(tag_vocab_path, attr_vocab_path, model_path):
    with open(tag_vocab_path, 'rb') as f:
        tag_vocab = pickle.load(f)
    with open(attr_vocab_path, 'rb') as f:
        attr_vocab = pickle.load(f)
    tokenizer = torchtext.data.utils.get_tokenizer('spacy')
    processor = DataProcessor(tag_vocab, attr_vocab, tokenizer, 128)
    model = build_model(tag_vocab, attr_vocab, model_path)
    return model, processor

# Predict the URL's nature
def predict_url(url, model, processor):
    pred, conf = run_model(url, model, processor)
    return pred, conf


def local_css(file_name):
    with open(file_name, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def main():
    local_css("style.css")  # Assuming you have a CSS file named style.css in the same directory

    st.title('URL Maliciousness Detector')
    st.markdown("""
    Enter a URL in the input field below to determine if it is malicious or benign. The analysis will indicate
    whether the provided URL is likely to lead to a malicious site, alongside a confidence score for the prediction.
    """, unsafe_allow_html=True)

    url = st.text_input("URL", placeholder="Enter URL here...")

    model, processor = load_resources('./data/tag_vocab.pkl', './data/attr_vocab.pkl', './models/model_65.pt')

    if st.button('Check URL'):
        with st.spinner('Analyzing...'):
            if url:
                pred, conf = predict_url(url, model, processor)
                if pred is None:
                    st.error(f"Cannot load or parse the website")
                elif pred == 1:
                    st.error(f"Malicious URL Detected with confidence {conf:.2%}")
                else:
                    st.success(f"URL is benign with confidence {conf:.2%}")


if __name__ == "__main__":
    main()
