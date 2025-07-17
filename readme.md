


# **EmotiCrafter: Text-to-Emotional-Image Generation based on Valence-Arousal Model** (ICCV2025)

Recent research shows that emotions can enhance users' cognition and influence information communication. While research on visual emotion analysis is extensive, limited work has been done on helping users generate emotionally rich image content.
Existing work on emotional image generation relies on discrete emotion categories, making it challenging to capture complex and subtle emotional nuances accurately. Additionally, these methods struggle to control the specific content of generated images based on text prompts.
In this paper, we introduce the task of continuous emotional image content generation (C-EICG) and present EmotiCrafter, a general emotional image generation model that generates images based on free text prompts and Valence-Arousal (V-A) values. It leverages a novel emotion-embedding mapping network to fuse V-A values into textual features, enabling the capture of emotions in alignment with intended input prompts. A novel loss function is also proposed to enhance emotion expression. The experimental results show that our method effectively generates images representing specific emotions with the desired content and outperforms existing techniques.

## 💡 What is EmotiCrafter?

We introduce the task of **Continuous Emotional Image Content Generation (C-EICG)** and present **EmotiCrafter**, a novel emotional image generation model that:

* Accepts **free-form text prompts**
* Conditions on **Valence-Arousal (V-A)** values
* Leverages a new **emotion-embedding mapping network** to fuse V-A signals into text features
* Uses a **custom loss function** to improve emotional fidelity

👉 [Try EmotiCrafter Demo on Hugging Face](https://huggingface.co/spaces/idvxlab/EmotiCrafter) 


## 🛠️ Setup Guide

### 1. Create Conda Environment

```bash
conda env create -f environment.yml
```

---

### 2. Clone the Repository

```bash
git clone https://github.com/idvxlab/EmotiCrafter
cd EmotiCrafter
```

---

### 3. Download the SDXL Base Model

You need to download the [Stable Diffusion XL Base 1.0 model](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) and place it appropriately.

---

### 4. Download Pretrained Model or Train Your Own

####  Option A: Use Pretrained Model (TBD)

Download the pretrained modal from [model](https://huggingface.co/spaces/idvxlab/EmotiCrafter/tree/main/model) and place it appropriately.

####  Option B: Train Your Own Model

##### a. Preprocess the Data

```bash
python preprocess.py --sdxl_path [pretrained SDXL]
```

##### b. Start Training

```bash
python train.py \
  --batch_size 768 \
  --lr 0.001 \
  --epochs 200 \
  --save_dir ./ckpt \
  --scale_factor 1.5 \
  --enable_density True
```

---

## 🖼️ Inference

Make sure you have your environment activated and model paths ready.

```bash
conda activate emotion
```

### Single Image Inference

```bash
python inference.py \
  --prompt "A man is running fast" \
  --arousal 2.5 \
  --valence -2 \
  --ckpt_path [pretrained_eit] \
  --sdxl_path  [pretrained_sdxl] \
  --seed 0
```

### 5x5 Grid Inference

```bash
python inference5x5.py \
  --prompt "A man is running fast" \
  --ckpt_path [pretrained_eit] \
  --sdxl_path  [pretrained_sdxl] \
  --seed 0
```



We thank the **[Stable Diffusion XL (SDXL)](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)**, **[FindingEmo](https://arxiv.org/abs/2402.01355)**, **[OASIS](https://link.springer.com/content/pdf/10.3758/s13428-016-0715-3.pdf)**, and **[Emotic](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w41/papers/Lapedriza_EMOTIC_Emotions_in_CVPR_2017_paper.pdf)** for the their excellent works, which made this work possible.

If you use **EmotiCrafter** in your research or applications, please cite our work.

