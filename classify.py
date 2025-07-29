import os
import shutil
import argparse
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModel
from tqdm import tqdm
from collections import Counter

# --- 1. Model Loading and Setup ---

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "openai/clip-vit-base-patch32"

processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

# --- 2. Folder and Prompt Setup ---
base_path = os.path.dirname(os.path.abspath(__file__))
REVIEW_MARGIN = 0.05  # If top-2 scores are within this, send to review

folders = {
    "indoor": {
        "few_shot": os.path.join(base_path, "few_shot/indoor"),
        "output": os.path.join(base_path, "classified/indoor"),
        "prompts": [
            "an indoor photo",
            "a photo taken indoor a building",
            "a room with furniture",
            "an image of an enclosed space",
            "a picture of a house interior"
        ]
    },
    "outdoor": {
        "few_shot": os.path.join(base_path, "few_shot/outdoor"),
        "output": os.path.join(base_path, "classified/outdoor"),
        "prompts": [
            "an outdoor photo",
            "a photo taken in nature",
            "a picture with trees and sky",
            "an image of the outdoor world",
            "a street scene"
        ]
    }
}

review_folder = os.path.join(base_path, "classified/review")
os.makedirs(review_folder, exist_ok=True)

# --- 3. Embedding Functions ---

def embed_image(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        image_emb = model.get_image_features(**inputs)
    return image_emb / image_emb.norm(dim=-1, keepdim=True)


def embed_prompt_list(prompts):
    inputs = processor(text=prompts, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        text_emb = model.get_text_features(**inputs)
    return (text_emb / text_emb.norm(dim=-1, keepdim=True)).mean(dim=0)


def embed_few_shot_images(folder):
    all_embeddings = []
    for fname in os.listdir(folder):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            try:
                emb = embed_image(os.path.join(folder, fname))
                all_embeddings.append(emb)
            except:
                print(f"Skipping broken image: {fname}")
    return torch.stack(all_embeddings).mean(dim=0) if all_embeddings else torch.zeros(512)


def initialize_class_prototypes():
    print("📐 Creating class prototypes...\n")
    for data in folders.values():
        data["prompt_embedding"] = embed_prompt_list(data["prompts"])
        data["image_embedding"] = embed_few_shot_images(data["few_shot"])

# --- 4. Classification Mode ---

def classify_unlabeled_images(unlabeled_folder):
    for data in folders.values():
        os.makedirs(data["output"], exist_ok=True)

    image_files = [f for f in os.listdir(unlabeled_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"📁 Classifying {len(image_files)} images...\n")

    for fname in tqdm(image_files):
        try:
            path = os.path.join(unlabeled_folder, fname)
            image_emb = embed_image(path).squeeze()

            scores = {}
            for label, data in folders.items():
                sim_text = torch.cosine_similarity(image_emb, data["prompt_embedding"], dim=0).item()
                sim_image = torch.cosine_similarity(image_emb, data["image_embedding"].squeeze(), dim=0).item()
                scores[label] = (sim_text + sim_image) / 2
            
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            top_label, top_score = sorted_scores[0]
            second_label, second_score = sorted_scores[1]

            if (top_score - second_score) < REVIEW_MARGIN:
                shutil.copy(path, os.path.join(review_folder, fname))
                decision = "REVIEW"
            else:
                shutil.copy(path, os.path.join(folders[top_label]["output"], fname))
                decision = top_label.upper()

            predicted_label = max(scores, key=scores.get)
            shutil.copy(path, os.path.join(folders[predicted_label]["output"], fname))

            print(f"{fname}: {top_label} ({top_score:.3f}) vs {second_label} ({second_score:.3f}) → {decision}")

        except Exception as e:
            print(f"❌ Failed on {fname}: {e}")

    print("\n✅ Classification complete.")

# --- 5. Validation Mode ---

def validate_classification_from_filenames(val_folder):
    image_files = [f for f in os.listdir(val_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"🧪 Validating {len(image_files)} images from '{val_folder}'...\n")

    y_true = []
    y_pred = []

    for fname in tqdm(image_files):
        try:
            path = os.path.join(val_folder, fname)
            image_emb = embed_image(path).squeeze()

            scores = {}
            for label, data in folders.items():
                sim_text = torch.cosine_similarity(image_emb, data["prompt_embedding"], dim=0).item()
                sim_image = torch.cosine_similarity(image_emb, data["image_embedding"].squeeze(), dim=0).item()
                scores[label] = (sim_text + sim_image) / 2

            predicted_label = max(scores, key=scores.get)
            y_pred.append(predicted_label)

            fname_lower = fname.lower()
            if fname_lower.startswith("indoor_"):
                y_true.append("indoor")
            elif fname_lower.startswith("outdoor_"):
                y_true.append("outdoor")
            else:
                print(f"⚠️ Skipping unknown labeled file: {fname}")
                continue

        except Exception as e:
            print(f"❌ Error on {fname}: {e}")
            continue

    correct = sum([yt == yp for yt, yp in zip(y_true, y_pred)])
    total = len(y_true)
    accuracy = correct / total if total > 0 else 0

    print(f"\n✅ Accuracy: {accuracy * 100:.2f}% ({correct}/{total})")

    print("\n📊 Confusion Matrix:")
    labels = ["indoor", "outdoor"]
    counts = Counter((t, p) for t, p in zip(y_true, y_pred))
    print(f"{'':10s}{'indoor':>10s}{'outdoor':>10s}")
    for t in labels:
        row = [counts.get((t, p), 0) for p in labels]
        print(f"{t:10s}{row[0]:>10d}{row[1]:>10d}")

# --- 6. Command-Line Interface ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify or validate images using hybrid CLIP similarity.")
    parser.add_argument("--mode", choices=["classify", "validate"], required=True, help="Choose 'classify' or 'validate'")
    parser.add_argument("--folder", type=str, required=True, help="Path to the folder containing images")

    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        print(f"❌ Error: Folder not found at '{args.folder}'")
    else:
        initialize_class_prototypes()

        if args.mode == "classify":
            classify_unlabeled_images(args.folder)
        elif args.mode == "validate":
            validate_classification_from_filenames(args.folder)
