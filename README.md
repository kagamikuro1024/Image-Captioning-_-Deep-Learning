# Image Captioning with Deep Learning
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Automatic Image Captioning** using Encoder-Decoder Architecture with CNN and LSTM/Attention Mechanisms

## 📑 Mục lục
- [Tổng quan](#-tổng-quan)
- [Dữ liệu](#-dữ-liệu)
- [Kiến trúc mô hình](#-kiến-trúc-mô-hình)
- [Kết quả](#-kết-quả)
- [Cài đặt](#-cài-đặt)
- [Sử dụng](#-sử-dụng)
- [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
- [Tham khảo](#-tham-khảo)

---

## 🎯 Tổng quan

Dự án này triển khai **bài toán sinh mô tả ảnh tự động (Image Captioning)** sử dụng kiến trúc **Encoder-Decoder**. Hệ thống có khả năng tự động tạo ra các câu mô tả bằng tiếng Anh cho hình ảnh đầu vào, kết hợp giữa:

- **Computer Vision**: Trích xuất đặc trưng hình ảnh
- **Natural Language Processing**: Sinh chuỗi văn bản mô tả

### Điểm nổi bật

✅ Hỗ trợ nhiều kiến trúc **CNN backbone** (ResNet50, EfficientNet B2-B4)  
✅ Triển khai **LSTM với Attention Mechanism**  
✅ Huấn luyện trên cả **Flickr8k** (8K ảnh) và **Flickr30k** (31K ảnh)  
✅ Đánh giá chi tiết với **BLEU Score** và **METEOR**  
✅ Hỗ trợ **Beam Search** và **Greedy Decoding**  
✅ **Transfer Learning** từ ImageNet pretrained models

---

## 📊 Dữ liệu

### Datasets

Dự án sử dụng hai bộ dữ liệu chuẩn:

| Dataset | Số ảnh | Số captions | Avg captions/image |
|---------|--------|-------------|-------------------|
| **Flickr8k** | 8,091 | 40,455 | 5 |
| **Flickr30k** | 31,783 | 158,915 | 5 |

### Cấu trúc dữ liệu

```
dataset/
├── Images/                # Thư mục chứa ảnh
│   └── flickr30k_images/
└── captions.txt          # File chứa captions (Flickr8k)
└── results.csv           # File chứa captions (Flickr30k)
```

### Định dạng Caption File

**Flickr8k** (`captions.txt`):
```
image,caption
1000268201_693b08cb0e.jpg,A child in a pink dress is climbing up a set of stairs in an entry way .
```

**Flickr30k** (`results.csv`):
```
image_name | comment_number | comment
1000092795.jpg | 0 | Two young guys with shaggy hair look at their hands while hanging out in the yard .
```

### Tiền xử lý dữ liệu

#### 1. Tiền xử lý ảnh (Image Preprocessing)

- **Resize**: Điều chỉnh kích thước theo yêu cầu của từng backbone
  - ResNet50: `224×224`
  - EfficientNet-B2: `260×260`
  - EfficientNet-B3: `300×300`
  - EfficientNet-B4: `380×380`
  
- **Normalization**: Chuẩn hóa theo ImageNet statistics
  ```python
  mean = [0.485, 0.456, 0.406]
  std = [0.229, 0.224, 0.225]
  ```

- **Feature Extraction**: Trích xuất feature maps từ CNN pretrained
  - ResNet50: `(49, 2048)` → 7×7 spatial grid
  - EfficientNet-B2: `(81, 1408)` → 9×9 spatial grid
  - EfficientNet-B3: `(100, 1536)` → 10×10 spatial grid
  - EfficientNet-B4: `(144, 1792)` → 12×12 spatial grid

#### 2. Tiền xử lý văn bản (Text Preprocessing)

- **Cleaning**:
  - Chuyển về chữ thường (lowercase)
  - Loại bỏ ký tự đặc biệt và số
  - Loại bỏ từ có độ dài ≤ 1

- **Special Tokens**:
  ```
  <PAD>: Padding token (index 0)
  <UNK>: Unknown words (index 1)
  startseq: Bắt đầu câu
  endseq: Kết thúc câu
  ```

- **Vocabulary Building**:
  - Flickr8k: 8,369 từ (min_freq=1)
  - Flickr30k: 20,157 từ (min_freq=2)
  - Max caption length: 34-40 từ

- **Padding**: Đưa tất cả sequences về cùng độ dài bằng `<PAD>` token

---

## 🏗️ Kiến trúc mô hình

### Tổng quan Encoder-Decoder

```
┌──────────────────────────────────────────────────────────────┐
│                     IMAGE CAPTIONING PIPELINE                 │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  [Input Image]                                                │
│       │                                                       │
│       ▼                                                       │
│  ┌─────────────────┐                                         │
│  │  CNN Encoder    │  (ResNet50 / EfficientNet)              │
│  │  - Pretrained   │                                         │
│  │  - Feature Maps │                                         │
│  └────────┬────────┘                                         │
│           │ Feature Vector (N×D)                             │
│           ▼                                                   │
│  ┌─────────────────┐                                         │
│  │  LSTM Decoder   │                                         │
│  │  + Attention    │                                         │
│  └────────┬────────┘                                         │
│           │                                                   │
│           ▼                                                   │
│  [Generated Caption]                                          │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### 1. Encoder: CNN Feature Extractor

#### ResNet50
- **Pretrained**: ImageNet (1000 classes)
- **Architecture**: 50 layers với residual connections
- **Output**: `(batch, 2048, 7, 7)` → Reshape to `(batch, 49, 2048)`
- **Params**: ~23M (frozen)

#### EfficientNet Family
- **Compound Scaling**: Tối ưu depth, width, resolution đồng thời
- **Variants**:

| Model | Input Size | Feature Map | Feature Dim | Params |
|-------|-----------|-------------|-------------|---------|
| EfficientNet-B2 | 260×260 | 9×9 | 1408 | ~7.7M |
| EfficientNet-B3 | 300×300 | 10×10 | 1536 | ~10.7M |
| EfficientNet-B4 | 380×380 | 12×12 | 1792 | ~17.7M |

### 2. Decoder: LSTM with Attention

#### Bahdanau Attention Mechanism

```python
# Attention computation tại mỗi timestep
α_t = softmax(v^T * tanh(W_encoder * encoder_out + W_decoder * h_{t-1}))
context_t = Σ(α_t * encoder_out)  # Weighted sum
```

**Ưu điểm**:
- Tập trung vào các vùng quan trọng của ảnh tại mỗi bước sinh từ
- Giải quyết vấn đề information bottleneck
- Cải thiện khả năng mô tả chi tiết

#### LSTM Architecture

```
┌─────────────────────────────────────────────────────┐
│              LSTM Decoder with Attention             │
├─────────────────────────────────────────────────────┤
│                                                      │
│  Encoder Features (N×D)                             │
│         │                                            │
│         ▼                                            │
│  ┌──────────────┐                                   │
│  │  Attention   │ ◄─── h_{t-1}                      │
│  └──────┬───────┘                                   │
│         │ context_t                                 │
│         ▼                                            │
│  ┌──────────────┐                                   │
│  │  Embedding   │ ◄─── word_{t-1}                   │
│  └──────┬───────┘                                   │
│         │                                            │
│         ▼                                            │
│  ┌──────────────┐                                   │
│  │  LSTMCell    │                                   │
│  │  h_t, c_t    │                                   │
│  └──────┬───────┘                                   │
│         │                                            │
│         ▼                                            │
│  ┌──────────────┐                                   │
│  │  Dropout     │                                   │
│  └──────┬───────┘                                   │
│         │                                            │
│         ▼                                            │
│  ┌──────────────┐                                   │
│  │  FC + Softmax│                                   │
│  └──────┬───────┘                                   │
│         │                                            │
│         ▼                                            │
│      word_t                                          │
│                                                      │
└─────────────────────────────────────────────────────┘
```

**Hyperparameters**:
```python
EMBED_SIZE = 512        # Word embedding dimension
HIDDEN_SIZE = 512       # LSTM hidden state size
ATTENTION_DIM = 512     # Attention layer dimension

EMBED_DROPOUT = 0.4     # Embedding dropout
LSTM_DROPOUT = 0.3      # LSTM dropout
DECODER_DROPOUT = 0.5   # Output dropout
```

### 3. Loss Function & Optimization

#### Loss Function
- **CrossEntropyLoss** với **Label Smoothing** (0.1)
- Bỏ qua `<PAD>` tokens trong tính toán loss

```python
criterion = nn.CrossEntropyLoss(
    ignore_index=0,      # Ignore <PAD>
    label_smoothing=0.1  # Reduce overconfidence
)
```

#### Optimizer
- **Adam** optimizer
- Learning rate: `1e-4` (Flickr8k), `3e-4` (Flickr30k)
- Weight decay: `1e-5`

#### Learning Rate Scheduler
- **ReduceLROnPlateau**
- Factor: 0.7
- Patience: 1-2 epochs
- Giảm LR khi validation loss không cải thiện

#### Early Stopping
- Patience: 5 epochs
- Dừng training khi validation loss không giảm

#### Gradient Clipping
- Clip norm: 5.0
- Tránh gradient explosion

---

## 📈 Kết quả

### So sánh các mô hình trên Flickr8k

| Model | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | Epochs |
|-------|--------|--------|--------|--------|---------|---------|
| **ResNet50 + LSTM + Attention** | 0.5166 | 0.3546 | 0.2372 | **0.1491** | - | 9 |
| **EfficientNet-B2 + LSTM + Attention** | 0.5061 | 0.3404 | 0.2273 | 0.1453 | 0.2949 | 12 |
| **EfficientNet-B3 + LSTM + Attention** | **0.5243** | **0.3540** | **0.2363** | 0.1507 | **0.3065** | 14 |
| ResNet50 + Transformer | 0.4807 | 0.3307 | 0.2207 | 0.1392 | - | - |

### Kết quả trên Flickr30k

| Model | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | Epochs |
|-------|--------|--------|--------|--------|---------|---------|
| **ResNet50 + LSTM + Attention** | 0.5034 | 0.3268 | 0.2149 | 0.1319 | 0.2662 | 22 |

### Đánh giá trên toàn bộ training set

| Model | Dataset | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR |
|-------|---------|--------|--------|--------|--------|---------|
| ResNet50 | Flickr8k | 0.5995 | 0.4425 | 0.3210 | 0.2227 | 0.3608 |
| ResNet50 | Flickr30k | 0.5746 | 0.4071 | 0.2870 | 0.1943 | 0.3213 |

### Nhận xét

#### ✅ Kết quả tốt nhất
**EfficientNet-B3 + LSTM + Attention** đạt hiệu suất tốt nhất trên Flickr8k:
- BLEU-1: **0.5243** (cao nhất)
- METEOR: **0.3065** (cao nhất)
- Cân bằng tốt giữa độ chính xác và khả năng tổng quát

#### 📊 Phân tích

1. **EfficientNet vs ResNet**:
   - EfficientNet-B3 vượt trội về BLEU-1 và METEOR
   - ResNet50 có BLEU-4 cao hơn một chút (0.1491 vs 0.1507)
   - EfficientNet hiệu quả hơn với số params ít hơn

2. **LSTM + Attention vs Transformer**:
   - LSTM + Attention vượt trội rõ rệt
   - Transformer đơn giản chưa đạt hiệu quả (cần thêm tricks)

3. **Flickr8k vs Flickr30k**:
   - Training trên Flickr8k cho kết quả test tốt hơn (overfitting ít hơn)
   - Flickr30k cần nhiều epochs và regularization hơn

### Ví dụ dự đoán

#### ✅ Trường hợp tốt
```
Image: 3066429707_842e50b8f7.jpg
Ground Truth: "girl in blue kicks the soccer ball"
Predicted: "girl in red shirt is playing soccer"
→ Nhận diện đúng: girl, playing soccer
```

#### ⚠️ Trường hợp cần cải thiện
```
Image: 476740978_45b65ebe0c.jpg
Ground Truth: "people holding pink signs that spell out impeach"
Predicted: "group of people stand on the street"
→ Thiếu chi tiết: signs, impeach
```

---

## 🔧 Cài đặt

### Yêu cầu hệ thống

- **OS**: Windows 10/11, Linux, macOS
- **Python**: 3.8+
- **GPU**: NVIDIA GPU với CUDA support (khuyến nghị)
  - RTX 3050 Ti 4GB: Batch size 12-32
  - RTX 3060 6GB+: Batch size 32-64
- **RAM**: 16GB+ (32GB khuyến nghị cho Flickr30k)
- **Storage**: 10GB+ free space

### Cài đặt dependencies

```bash
# Clone repository
git clone https://github.com/yourusername/image-captioning.git
cd image-captioning

# Tạo virtual environment (khuyến nghị)
conda create -n image_caption python=3.9
conda activate image_caption

# Hoặc dùng venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Cài đặt PyTorch với CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Cài đặt các thư viện khác
pip install -r requirements.txt
```

### requirements.txt
```txt
torch>=2.0.0
torchvision>=0.15.0
Pillow>=9.0.0
numpy>=1.21.0
tqdm>=4.62.0
nltk>=3.6.0
tensorboard>=2.11.0
```

### Download datasets

#### Flickr8k
```bash
# Download from Kaggle
# https://www.kaggle.com/datasets/adityajn105/flickr8k

# Hoặc dùng Kaggle API
kaggle datasets download -d adityajn105/flickr8k
unzip flickr8k.zip -d content/clean_data_flickr8k/
```

#### Flickr30k
```bash
# Download from Kaggle
# https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset

kaggle datasets download -d hsankesara/flickr-image-dataset
unzip flickr-image-dataset.zip -d content/clean_data_flickr30k/
```

### Download pretrained models (Optional)

```bash
# Download best model weights
wget https://your-link/best_model_efficientnet_b3_h512.pth -P content/clean_data_flickr8k/
```

---

## 🚀 Sử dụng

### 1. Trích xuất features (Lần đầu tiên)

```bash
# Với ResNet50
python ResNet50_LSTM_Attention/ResNet50_LSTM_Attention_Good.py --mode extract

# Với EfficientNet-B3
python EfficientNet_LSTM_Attention/EfficientNet_Full_Flavor_LSTM_Attention.py --mode extract
```

**Lưu ý**: Quá trình này sẽ tạo file `.pkl` chứa features (có thể lớn hơn 4GB):
- ResNet50 Flickr8k: ~4GB
- EfficientNet-B3 Flickr8k: ~4.7GB
- EfficientNet-B3 Flickr30k: ~18GB

### 2. Training

#### ResNet50 + LSTM + Attention (Flickr8k)

```bash
cd ResNet50_LSTM_Attention
python ResNet50_LSTM_Attention_Good.py --mode train
```

**Thời gian training**: ~5 phút/epoch trên RTX 3050 Ti (30 epochs = ~2.5 giờ)

#### EfficientNet-B3 + LSTM + Attention (Flickr8k)

```bash
cd EfficientNet_LSTM_Attention

# Chỉnh sửa Config trong file .py:
# EFFICIENTNET_VARIANT = 'b3'  # Chọn 'b2', 'b3', or 'b4'
# BATCH_SIZE = 32              # Điều chỉnh theo VRAM

python EfficientNet_Full_Flavor_LSTM_Attention.py --mode train
```

**Thời gian training**: ~4-5 phút/epoch (40 epochs = ~3 giờ)

#### ResNet50 + LSTM + Attention (Flickr30k)

```bash
cd ResNet50_LSTM_Attention
python ResNet50_LSTM_Attention_Good.py --mode train

# Hoặc chạy riêng cho Flickr30k
python ResNet50_LSTM_Attention_Flickr30k.py --mode train
```

**Thời gian training**: ~23-25 phút/epoch (50 epochs = ~20 giờ)

### 3. Evaluation

```bash
# Đánh giá trên test set
python ResNet50_LSTM_Attention_Good.py --mode eval

# Đánh giá chi tiết với METEOR
python ResNet50_LSTM_Attention_Good.py --mode research_eval

# Xem TensorBoard logs
tensorboard --logdir=content/clean_data_flickr8k/runs
```

### 4. Inference - Sinh caption cho ảnh mới

```python
from PIL import Image
from EfficientNet_Full_Flavor_LSTM_Attention import *

# Load model
config = Config()
model = load_model(config)

# Load và preprocess ảnh
image = Image.open('path/to/your/image.jpg')

# Generate caption với Greedy Decoding
caption_greedy = generate_caption_greedy(model, image, vocab)
print(f"Greedy: {caption_greedy}")

# Generate caption với Beam Search (tốt hơn)
caption_beam = generate_caption_beam(model, image, vocab, beam_size=3)
print(f"Beam Search (k=3): {caption_beam}")
```

### 5. Testing script

```bash
# Test với một ảnh cụ thể
cd EfficientNet_LSTM_Attention
python Test_Image_Caption.py --image path/to/image.jpg --model b3
```

---

## 📁 Cấu trúc thư mục

```
image-captioning/
│
├── EfficientNet_LSTM_Attention/          # EfficientNet models
│   ├── EfficientNet_Full_Flavor_LSTM_Attention.py  # Main training script
│   ├── EfficientNet_Kaggle.ipynb        # Jupyter notebook version
│   ├── Test_Image_Caption.py            # Inference script
│   ├── check_dataset.py                 # Dataset verification
│   ├── requirements.txt                 # Dependencies
│   ├── EfficientNetB2_LSTM_Attention.txt  # B2 training log
│   ├── EfficientNetB3_LSTM_Attention.txt  # B3 training log
│   ├── EfficientNetB4_LSTM_Attention.txt  # B4 training log
│   ├── best_model_efficientnet_b3_h512.pth  # Best model weights
│   └── content/
│       └── clean_data_flickr30k/        # Flickr30k dataset
│           ├── Images/
│           ├── results.csv
│           ├── features_efficientnet_b3.pkl
│           └── runs/                    # TensorBoard logs
│
├── ResNet50_LSTM_Attention/              # ResNet50 models
│   ├── ResNet50_LSTM_Attention_Good.py  # Main script (Flickr8k)
│   ├── ResNet50_LSTM_Attention.ipynb    # Jupyter notebook
│   ├── ResNet50_LSTM_Attention.txt      # Flickr8k training log
│   ├── ResNet50_30k.txt                 # Flickr30k training log
│   └── Dec21_*/                         # Training runs
│
├── ResNet50_Transformer_Simple/          # Transformer experiments
│   ├── ResNet50_Transformer.py
│   ├── ResNet50_Transformer.ipynb
│   └── ResNet50_Transformer.txt
│
├── Báo cáo Học sâu/                      # LaTeX report
│   ├── tr21-60.tex                      # Main LaTeX file
│   ├── svmult.cls                       # Document class
│   └── chapters/                        # Report chapters
│
├── Caption-Normalization-Section.ipynb   # Data preprocessing
├── ImageCaptioning.ipynb                 # Overview notebook
├── README.md                             # This file
└── requirements.txt                      # Global dependencies
```

---

## 📚 Tham khảo

### Papers

1. **Show, Attend and Tell** (Xu et al., 2015)
   - Attention mechanism for image captioning
   - [arXiv:1502.03044](https://arxiv.org/abs/1502.03044)

2. **Deep Residual Learning** (He et al., 2016)
   - ResNet architecture
   - [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)

3. **EfficientNet: Rethinking Model Scaling** (Tan & Le, 2019)
   - EfficientNet compound scaling
   - [arXiv:1905.11946](https://arxiv.org/abs/1905.11946)

4. **BLEU: a Method for Automatic Evaluation** (Papineni et al., 2002)
   - BLEU score metric

5. **METEOR: An Automatic Metric for MT Evaluation** (Banerjee & Lavie, 2005)
   - METEOR score metric

### Datasets

- **Flickr8k**: [Kaggle Dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k)
- **Flickr30k**: [Kaggle Dataset](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset)

### Code References

- PyTorch Official Tutorials
- [Show and Tell Implementation](https://github.com/yunjey/pytorch-tutorial)

---

## 🎓 Contributors

- **Tạ Quốc Tuấn** - Team Lead
- **Phan Trọng Đức** - Architecture Design
- **Đoàn Ngọc Toàn** - Implementation
- **Lê Văn Quang Trung** - Evaluation

**Trường Đại học Công nghệ Thông tin - ĐHQG TP.HCM**

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Đề tài bài tập lớn môn **Học Sâu (Deep Learning)**
- Cảm ơn Kaggle và cộng đồng open-source
- Pretrained models từ PyTorch Model Zoo

---

## 📧 Contact

Nếu có câu hỏi hoặc góp ý, vui lòng liên hệ:
- Email: [your.email@example.com]
- Issues: [GitHub Issues](https://github.com/yourusername/image-captioning/issues)

---

**⭐ Nếu thấy dự án hữu ích, hãy cho chúng tôi một star!**
