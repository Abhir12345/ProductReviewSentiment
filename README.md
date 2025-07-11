# 🧵 Fashion Product Review Sentiment Analyzer

This project uses a fine-tuned DistilBERT model to classify customer reviews of fashion products as **Positive** or **Negative**. The model was trained on real-world e-commerce data and deployed using **Gradio** on Hugging Face Spaces.

<div align="center">
  <img src="https://img.shields.io/badge/Model-DistilBERT-blueviolet" />
  <img src="https://img.shields.io/badge/Epochs-3-success" />
  <img src="https://img.shields.io/badge/Accuracy-~92%25-brightgreen" />
  <img src="https://img.shields.io/badge/Live-Demo-ff69b4" />
</div>

---

## 🚀 Live Demo

👉 [Click here to try it out!](https://abhinandan12345-product-sentiment-review.hf.space/?__theme=system&deep_link=fBSSVo8rs3s)

---

## 📊 Dataset

- [🛍️ Women's E-Commerce Clothing Reviews (Kaggle)](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews)
- Contains ~23,000 clothing product reviews with associated ratings, text, and metadata.

---

## 🧠 Model Details

- **Model Architecture**: `DistilBERTForSequenceClassification`
- **Pretrained Base**: [`distilbert-base-uncased`](https://huggingface.co/distilbert-base-uncased)
- **Fine-Tuned On**: Labeled sentiment (rating-based) from fashion product reviews
- **Number of Epochs**: 3
- **Classification**: Binary (Positive / Negative)
- **Achieved Accuracy**: ~92%

---

## 🛠️ Tech Stack

| Component       | Tool/Library                   |
|----------------|--------------------------------|
| Language        | Python                         |
| Model           | Hugging Face Transformers      |
| Tokenizer       | `DistilBertTokenizerFast`      |
| Inference       | Manual using PyTorch           |
| Deployment      | Gradio + Hugging Face Spaces   |

---

## 💡 How It Works

1. **Input**: User enters a raw fashion product review.
2. **Tokenizer**: Converts text to BERT-compatible tokens.
3. **Model**: Predicts sentiment (0 = Negative, 1 = Positive).
4. **Output**: Sentiment label with confidence score.

---

## 🧪 Example

> _"The dress fits beautifully and the fabric is amazing!"_

**Prediction**: `Positive (97.21%)`

---

## 📂 Project Structure
-  E_commerge.ipynb # The notebook used for training, and text cleaning. 
- 🔐 Model Weights
- *Due to GitHub file **size limits**, model weights are hosted on the Hugging Face Hub*:

👉 [Abhinandan12345/distilbert-fashion-sentiment](https://huggingface.co/Abhinandan12345/distilbert-fashion-sentiment)
-  gradio_app.py # Main Gradio UI script
-  equirements.txt # Python dependencies
-  README.md


---

## 📌 To Run Locally

```bash
git clone https://github.com/yourusername/fashion-sentiment-analyzer.git
cd fashion-sentiment-analyzer
pip install -r requirements.txt
python gradio_app.py

