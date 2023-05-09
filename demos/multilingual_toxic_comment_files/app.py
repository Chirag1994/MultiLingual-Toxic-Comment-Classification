import tensorflow as tf
import gradio as gr
import pandas as pd
from transformers import AutoTokenizer

model_save_path = "Multilingual_toxic_comment_classifier/"
### Loading the fine-tuned model ###
loaded_model = tf.keras.models.load_model(model_save_path)
### Initializing the tokenizer ###
tokenizer_ = AutoTokenizer.from_pretrained("xlm-roberta-large")

examples_list = [
    [example]
    for example in pd.read_csv("examples/sample_comments.csv")["comment_text"].tolist()
]


def prep_data(text, tokenizer, max_len=192):
    tokens = tokenizer(
        text,
        max_length=max_len,
        truncation=True,
        padding="max_length",
        add_special_tokens=True,
        return_tensors="tf",
    )

    return {
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"],
    }


def predict(text):
    prob_of_toxic_comment = loaded_model.predict(
        prep_data(text=text, tokenizer=tokenizer_, max_len=192)
    )[0][0]
    prob_of_non_toxic_comment = 1 - prob_of_toxic_comment
    prob_of_toxic_comment, prob_of_non_toxic_comment
    probs = {
        "prob_of_toxic_comment": float(prob_of_toxic_comment),
        "prob_of_non_toxic_comment": float(prob_of_non_toxic_comment),
    }
    return probs


interface = gr.Interface(
    fn=predict,
    inputs=gr.components.Textbox(lines=4, label="Comment"),
    outputs=[gr.Label(label="Probabilities")],
    examples=examples_list,
    title="Multi-Lingual Toxic Comment Classification.",
    description="XLM-Roberta Large model",
)
interface.launch(debug=False, share=True)
