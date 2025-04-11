# /// script
# requires-python = "==3.12"
# dependencies = [
#     "torch",
#     "transformers",
#     "model2vec",
#     "sentence-transformers",
# ]
# ///


import warnings
from transformers import AutoTokenizer, AutoModel
from transformers import CLIPModel
from transformers import logging
from model2vec import StaticModel
from sentence_transformers import SentenceTransformer
import argparse


def embeddings(model_dir, text):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModel.from_pretrained(model_dir, local_files_only=True)
    encoded_input = tokenizer(text, return_tensors="pt")
    output = model(**encoded_input)
    return output[0][:, 0, :].flatten().tolist()


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


def main(model_dir, text, emb_type="bert"):
    if emb_type == "bert" or emb_type == "xlm-roberta" or emb_type == "roberta":
        values = embeddings(model_dir, text)
    elif emb_type == "clip":
        values = clip_embeddings(model_dir, text)
    elif emb_type == "model2vec":
        values = model2vec_embeddings(model_dir, text)
    elif emb_type == "static-embeddings":
        values = static_embeddings(model_dir, text)
    else:
        raise ValueError(f"Unknown emb_type: {emb_type}")
    print("\n".join([str(x) for x in values]))


# run e.g: `uv run generate.py "./cache/google-bert/bert-base-uncased" "Text to encode"` bert
if __name__ == "__main__":
    logging.set_verbosity_error()
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=str, help="Model local dir")
    parser.add_argument("text", type=str, help="Text to embed")
    parser.add_argument("type", type=str, help="Embedding type")
    args = parser.parse_args()
    main(args.model_dir, args.text, args.type)
