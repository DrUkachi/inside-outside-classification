# 🏠🏞 Indoor/Outdoor Image Classification

This project contains a Python script for classifying images as either **indoor** or **outdoor** scenes using a pre-trained CLIP model and a lightweight few-shot, prototype-based method. It was built in response to a practical classification task using real-world, unlabeled image data.

---

## 📦 Description

The classifier leverages **OpenAI’s CLIP (`clip-vit-base-patch32`)** as a visual encoder and classifies new images by comparing their embeddings to **representative class prototypes** created from a small set of hand-picked example images (few-shot learning).

It is:

* ✅ **Training-free** (no fine-tuning needed)
* ⚡ **Fast**, with support for CPU
* 🔧 **Modular**, and easy to improve further

---

## ⚙️ Requirements

* Python 3.8+
* Dependencies listed in `requirements.txt`

### 🔧 Install

```bash
pip install -r requirements.txt
```

This installs:

* PyTorch
* HuggingFace Transformers
* Pillow (for image handling)
* tqdm (for progress bars)

---

## 🚀 How to Run

### 1. Prepare Your Directory

Your project structure should look like this:

```
few_shot/
├── indoor/         # 14 indoor example images
└── outdoor/        # 56 outdoor example images
unlabeled/          # images to classify
validation/         # images used for validation
classified/
├── indoor/
├── outdoor/
└── review/         # 🔄 images that are ambiguous or borderline
```

### 📂 Optional: Auto-Unzip ZIP Files

The `few_shot`, `validation`, and `unlabeled` image folders are stored as ZIP files in the `data/` folder (e.g., `data/few_shot.zip`), the script will automatically unzip them into the root directory when it runs.

**Expected ZIP locations:**

```
data/
├── few_shot.zip
├── validation.zip
└── unlabeled.zip
```

When you run the script, it will unzip these files into:

```
few_shot/
validation/
unlabeled/
```

> 💡 The folders will only be unzipped if they **don’t already exist**, so you can safely rerun the script.


### 2. Example Usage

#### I. To classify images:

```bash
python classify.py --mode classify --folder ./unlabeled
```

#### II. To validate predictions using filename labels:

```bash
python classify.py --mode validate --folder ./validation
```

All classified images will be sorted into `classified/indoor/`, `classified/outdoor/`, or `classified/review/` folders.

---

## 🧠 Methodology

### ✔ Model

* Uses `openai/clip-vit-base-patch32` from HuggingFace
* Embeddings are extracted for both:

  * **Text prompts** (e.g., “a photo taken indoors”)
  * **Few-shot example images**

### ✔ Classification Logic

* For each image:

  1. Get its CLIP embedding
  2. Compare it to the **averaged prompt embeddings** and **few-shot image embeddings**
  3. Compute similarity to both classes
  4. 🔄 If the top two scores are close (within 0.05), the image is **sent to review**
  5. Otherwise, assign to the class with the highest similarity

---

## ⚖️ What Does the `REVIEW_MARGIN = 0.05` Mean?

When the model is **unsure** (i.e., the similarity difference between the top two classes is small), the image is routed to a **`review/` folder** for manual inspection.

> 🧠 *Why 0.05?*
> A 5% margin was selected as a **practical threshold for ambiguity**. It captures borderline cases where CLIP's semantic similarity doesn't clearly favor one class. This value can be tuned depending on tolerance for false positives or the capacity for human review.

---

## 🧪 Observations & Insights

### ✅ Strengths

* No model training required
* Strong generalization from a few visual examples
* Prompt-based reasoning makes it adaptable to other classes

### ⚠️ Limitations & Edge Cases

| Category                      | Example                             | Insight                                                                             |
| ----------------------------- | ----------------------------------- | ----------------------------------------------------------------------------------- |
| **Ambiguous scenes**          | `190881191_*.jpg`                   | Even humans disagree — routed to **review/** folder                                 |
| **Roofed or car interiors**   | `219636488_*.jpg`, `70939958_*.jpg` | Challenging without contextual metadata                                             |
| **Clear misclassifications**  | `79869777_*.jpg`, `227589596_*.jpg` | Could benefit from a secondary model (e.g., object detection)                       |
| **Unexplainable predictions** | `253900795_*.jpg`                   | Explaining CLIP decisions is non-trivial — visual interpretability tools could help |
| **Prompt sensitivity**        | `50587842_*.jpg`, `56540294_*.jpg`  | Slight changes in text can impact results — consider dynamic prompt ensembling      |
| **Environmental cues**        | `99454779_*.jpg`                    | Brightness, lighting, and framing may bias CLIP's perception                        |

---

## 💡 What I'd Improve with More Time

1. **Scene-Based Inference Engine**
   Use models like **Places365** to classify contextually confusing cases (e.g., parking lots, stadiums).

2. **Explainability Tools**
   Add **SHAP**, **Grad-CAM**, or embedding heatmaps to interpret classification decisions.

3. **Prompt Augmentation & Tuning**
   Dynamically improve text prompts using automated selection or fine-tuned language prompts.

---

## 📁 Deliverables

* `classify.py`: Main classification and validation script
* `README.md`: Full guide and technical report (this file)
* Folder structure with `few_shot`, `unlabeled`, `validation`, `review`, and `classified/` directories

---

## 🏁 Evaluation Criteria

This solution demonstrates:

* ✅ Effective use of large pre-trained vision-language models
* ✅ Lightweight and reproducible code
* ✅ Clear handling of edge cases
* ✅ Review mechanism for ambiguous images
* ✅ Good modularity for future extensions