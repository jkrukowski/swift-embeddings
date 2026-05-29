# /// script
# requires-python = "==3.12"
# dependencies = [
#     "torch",
#     "transformers",
#     "model2vec>=0.5.0",
#     "sentence-transformers",
#     "einops",
# ]
# ///


import warnings
import torch
from transformers import AutoTokenizer, AutoModel
from transformers import CLIPModel
from transformers import logging
from model2vec import StaticModel
from sentence_transformers import SentenceTransformer
import argparse


def modern_bert(model_dir, text):
    model = SentenceTransformer(model_dir, truncate_dim=768)
    output = model.encode(text)
    return output.flatten().tolist()


def embeddings(model_dir, text):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModel.from_pretrained(model_dir, local_files_only=True)
    if isinstance(text, str):
        encoded_input = tokenizer(text, return_tensors="pt")
        output = model(**encoded_input)
        return output[0][:, 0, :].flatten().tolist()
    elif isinstance(text, list):
        encoded_input = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        output = model(**encoded_input)
        return output.last_hidden_state[:, 0, :].flatten().tolist()

def clip_embeddings(model_dir, text):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = CLIPModel.from_pretrained(model_dir, local_files_only=True)
    encoded_input = tokenizer(text, return_tensors="pt")
    output = model.text_model(**encoded_input)
    return output.pooler_output.flatten().tolist()


def model2vec_embeddings(model_dir, text):
    model = StaticModel.from_pretrained(model_dir)
    output = model.encode(text)
    return output.flatten().tolist()


def static_embeddings(model_dir, text):
    model = SentenceTransformer(model_dir, truncate_dim=1023)
    output = model.encode(text, normalize_embeddings=True)
    return output.flatten().tolist()

def nomic_embeddings(model_dir, text):
    model = SentenceTransformer(model_dir, truncate_dim=768, trust_remote_code=True)
    output = model.encode(text, normalize_embeddings=False)
    return output.flatten().tolist()


def qwen3_embeddings(model_dir, text):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    tokenizer.padding_side = "right"
    model = AutoModel.from_pretrained(model_dir, local_files_only=True)
    if isinstance(text, str):
        text = [text]
    encoded_input = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    with torch.no_grad():
        output = model(**encoded_input)
    last_hidden = output.last_hidden_state  # [batch, seq, hidden]
    attention_mask = encoded_input["attention_mask"]
    # right padding -> last non-padding index is sum(mask) - 1
    last_indices = attention_mask.sum(dim=1) - 1
    pooled = last_hidden[torch.arange(last_hidden.size(0)), last_indices]
    pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
    return pooled.flatten().tolist()


def main(model_dir, text, emb_type="bert"):
    if emb_type == "bert" or emb_type == "xlm-roberta" or emb_type == "roberta":
        values = embeddings(model_dir, text)
    elif emb_type == "modernbert":
        values = modern_bert(model_dir, text)
    elif emb_type == "clip":
        values = clip_embeddings(model_dir, text)
    elif emb_type == "model2vec":
        values = model2vec_embeddings(model_dir, text)
    elif emb_type == "static-embeddings":
        values = static_embeddings(model_dir, text)
    elif emb_type == "nomic":
        values = nomic_embeddings(model_dir, text)
    elif emb_type == "qwen3":
        values = qwen3_embeddings(model_dir, text)
    else:
        raise ValueError(f"Unknown emb_type: {emb_type}")
    print("\n".join([str(x) for x in values]))


# run e.g: `uv run generate.py --model_dir "./cache/google-bert/bert-base-uncased" --text "Text to encode" --type bert`
if __name__ == "__main__":
    logging.set_verbosity_error()
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, help="Model local dir")
    parser.add_argument("--text", action='append', help="Text to embed")
    parser.add_argument("--type", type=str, help="Embedding type")
    args = parser.parse_args()
    text_input = args.text[0] if len(args.text) == 1 else args.text
    main(args.model_dir, text_input, args.type)
