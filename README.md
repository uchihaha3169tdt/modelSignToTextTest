"# 🤘 YouTube Sign Language Generation

Text-to-Sign-Language motion generation using diffusion models, customized for Vietnamese sign language from YouTube videos.

## 📋 Overview

This repository is adapted from [Sign-Diffusion-Model](https://github.com/kha-kim-thuy/Sign-Diffusion-Model) to work with sign language data extracted from YouTube videos using MediaPipe and OCR.

**Key Features:**
- ✅ Support for OpenPose keypoints (50 joints, 150 dimensions)
- ✅ Vietnamese and English text captions
- ✅ Space-Time U-Net with Scale-Aware Modulation (SAM-STUNet)
- ✅ Diffusion-based motion generation

## 🔧 Installation

```bash
# Clone repository
git clone https://github.com/uchihaha3169tdt/modelSignToTextTest.git
cd modelSignToTextTest

# Setup environment
bash scripts/setup.sh
```

## 📁 Dataset Structure

Place your processed YouTube Sign dataset in:

```
dataset/YOUTUBE_SIGN/
├── new_joints/         # .npy motion files [T, 150]
├── texts/              # .txt caption files
├── train.txt           # training split
├── val.txt             # validation split
├── test.txt            # test split
└── all.txt             # all samples
```

**Motion format:** OpenPose keypoints with 50 joints × 3 coordinates = 150 dimensions per frame.

**Text format:** Each `.txt` file contains lines in format:
```
caption#token1/POS token2/POS ...#0.0#0.0
```

## 🚀 Training

### 1. Prepare data

```bash
bash scripts/prepare_data.sh
```

This will calculate Mean.npy and Std.npy statistics for normalization.

### 2. Download GloVe embeddings

```bash
bash prepare/download_glove.sh
```

### 3. Train the model

```bash
bash scripts/train_youtube_sign.sh
```

Or manually:

```bash
python -m train.train_mdm \
    --arch sam_stunet \
    --lr 1e-4 \
    --overwrite \
    --save_interval 1000 \
    --num_steps 400000 \
    --dataset youtube_sign \
    --save_dir ./save/youtube_sign_model \
    --batch_size 64 \
    --diffusion_steps 1000 \
    --device 0
```

## 📊 Model Architecture

The main model is **SAM-STUNet** (Space-Time U-Net with Scale-Aware Modulation):
- Encoder-decoder U-Net architecture for temporal modeling
- Scale-aware modulation for multi-scale feature fusion
- Classifier-free guidance for text conditioning
- Multilingual CLIP (clip-ViT-B-32-multilingual-v1) for text encoding

## 📂 Repository Structure

```
├── data_loaders/
│   ├── get_data.py          # Dataset loader factory (supports youtube_sign)
│   ├── tensors.py           # Tensor utilities and collate functions
│   └── humanml/
│       ├── data/dataset.py  # Dataset classes including YouTubeSign
│       └── utils/
│           ├── get_opt.py   # Dataset configuration (includes youtube_sign)
│           ├── word_vectorizer.py
│           └── metrics.py
├── diffusion/               # Diffusion model utilities
├── model/
│   ├── mdm.py               # MDM transformer model
│   └── sam_stunet.py        # SAM-STUNet architecture (main model)
├── train/                   # Training scripts
├── sample/generate.py       # Generation/sampling script
├── utils/                   # Utility functions
├── prepare/
│   ├── calculate_stats.py   # Calculate dataset statistics
│   └── download_glove.sh    # Download GloVe embeddings
├── dataset/
│   └── youtube_sign_opt.txt # Dataset configuration
└── scripts/
    ├── setup.sh             # Environment setup
    ├── prepare_data.sh      # Data preparation
    └── train_youtube_sign.sh # Training script
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Original [Sign-Diffusion-Model](https://github.com/kha-kim-thuy/Sign-Diffusion-Model) by kha-kim-thuy
- [MDM: Human Motion Diffusion Model](https://github.com/GuyTevet/motion-diffusion-model)
- [text-to-motion](https://github.com/EricGuo5513/text-to-motion)
" 
