# 🔬 Malaria Cell Image Classification using VGG19 & Grey Wolf Optimizer (GWO)

> **Laporan Eksperimen & Implementasi Transfer Learning Deep Learning dengan Metaheuristik Hyperparameter Tuning**

---

## 📌 Ringkasan Eksekutif

Proyek ini berfokus pada pengembangan sistem klasifikasi citra medis otomatis untuk membedakan sel darah terinfeksi malaria (*Parasitized*) dan sel darah sehat (*Uninfected*). Model dibangun mengintegrasikan arsitektur **VGG19 (Transfer Learning)** dan algoritma metaheuristik **Grey Wolf Optimizer (GWO)** via library `mealpy` untuk melakukan pencarian kombinasi hyperparameter optimal secara otomatis.

Dalam pengujian pada **5.512 citra uji (Test Set)**, model hasil optimasi GWO berhasil mencapai tingkat akurasi sebesar **96.03%** dengan evaluasi *loss* sebesar **0.1188**.

---

## 🎯 Latar Belakang & Tujuan

Diagnosis mikroskopis manual terhadap sampel apusan darah tebal/tipis memerlukan ketelitian tinggi dan memakan waktu. Penerapan *Convolutional Neural Networks* (CNN) terbukti efektif dalam analisis citra medis, namun performa model sangat bergantung pada pemilihan hyperparameter (*learning rate*, *dropout*, arsitektur *fully connected layer*, dan strategi *fine-tuning*).

**Tujuan Utama Proyek:**
1. Mengembangkan skema klasifikasi sel darah malaria otomatis berbasis VGG19.
2. Memanfaatkan algoritma **Grey Wolf Optimizer (GWO)** untuk mengeksplorasi dan mengeksploitasi *search space* hyperparameter tanpa *trial-and-error* manual.
3. Mengukur efektivitas model akhir berdasarkan metrik akurasi, *loss*, *precision*, *recall*, dan *F1-Score*.

---

## 📊 Dataset & Pra-pemrosesan Data

Dataset tidak diunggah ke repositori ini karena ukurannya yang besar. Anda dapat mengunduh dataset resmi di:
- [Kaggle - Malaria Cell Images Dataset](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria)

### 1. Pembagian Data (Data Splitting)
Data dibagi secara *Stratified Random Sampling* untuk menjaga keseimbangan rasio kelas pada setiap bagian:

| Subset Data | Persentase | Jumlah Citra | Distribusi Kelas |
|---|---|---|---|
| **Training Set** | 64% | 17.636 | 8.818 Parasitized / 8.818 Uninfected |
| **Validation Set** | 16% | 4.410 | 2.205 Parasitized / 2.205 Uninfected |
| **Test Set** | 20% | 5.512 | 2.756 Parasitized / 2.756 Uninfected |
| **Total** | **100%** | **27.558** | **13.779 Parasitized / 13.779 Uninfected** |

### 2. Pipeline Pipeline Preprocessing & Augmentasi
- **Image Resizing**: Seluruh citra diubah ukurannya menjadi dimensi seragam $128 \times 128 \times 3$ piksel (RGB).
- **Normalization**: Rescaling nilai piksel dari rentang $[0, 255]$ ke $[0.0, 1.0]$.
- **Data Augmentation**: Diterapkan pada *Training Set* menggunakan `ImageDataGenerator` untuk mencegah *overfitting*:
  - Rotation Range: $20^\circ$
  - Width & Height Shift: $15\%$
  - Shear Range: $15\%$
  - Zoom Range: $15\%$
  - Horizontal Flip: Enabled

---

## ⚙️ Metodologi & Optimasi GWO

### 1. Arsitektur Base Model
- **Backbone**: VGG19 *pre-trained* pada ImageNet.
- **Classification Head**: *Global Average Pooling 2D* / *Flatten* $\rightarrow$ *Dense Layer* (ReLU) $\rightarrow$ *Dropout* $\rightarrow$ *Dense Output Layer* (Softmax/Sigmoid).

### 2. Ruang Pencarian Hyperparameter (GWO Search Space)
Algoritma GWO memodelkan kepemimpinan hirarki dan mekanisme perburuan kawanan serigala abu-abu ($\alpha, \beta, \delta$) untuk mencari solusi optimal dalam batasan berikut:

| Hyperparameter | Tipe Data | Batas Minimum / Pilihan | Batas Maksimum / Pilihan |
|---|---|---|---|
| `learning_rate` | Continuous | $1 \times 10^{-5}$ ($0.00001$) | $1 \times 10^{-3}$ ($0.001$) |
| `dropout_rate` | Continuous | $0.20$ | $0.70$ |
| `dense_units` | Discrete | $128$ | $512$ |
| `unfreeze_layers` | Discrete | $0$ (Frozen) | $15$ (Fine-tune Block 4 & 5) |

---

## 🏆 Hasil Optimasi & Konfigurasi Terbaik

Berdasarkan iterasi konvergensi algoritma Grey Wolf Optimizer, diperoleh kombinasi hyperparameter terbaik sebagai berikut:

| Parameter | Hasil Optimasi GWO | Keterangan Eksekusi |
|---|---|---|
| **Optimal Learning Rate** | `0.000100` (`1e-4`) | Menggunakan Optimizer Adam |
| **Optimal Dropout Rate** | `0.494` | Regulerisasi pada Fully Connected Layer |
| **Dense Units** | `256` | Jumlah neuron pada penutup klasifikasi |
| **Unfreeze Layers** | `5` | Membuka pelatihan pada 5 layer teratas (Block 5 VGG19) |

---

## 📈 Evaluasi & Performa Model

Pengujian model final dilakukan terhadap **5.512 sampel data uji (Test Set)** yang tidak pernah dilihat oleh model selama proses training maupun pencarian hyperparameter.

### Metrik Performa Utama

| Metrik Evaluasi | Nilai Capaian |
|---|---|
| **Test Accuracy** | **96.03%** |
| **Test Loss** | **0.1188** |
| **Precision (Parasitized)** | $\approx 96.1\%$ |
| **Recall (Parasitized)** | $\approx 95.9\%$ |
| **F1-Score** | **0.96** |

---

## 📁 Struktur Repositori

```text
malaria-vgg19-gwo/
├── .gitignore                  # Berkas pengecualian Git (dataset, cache, weights)
├── README.md                   # Dokumentasi dan laporan proyek
├── requirements.txt            # Daftar pustaka & dependensi Python
├── best_hyperparameters.json   # Output parameter hasil pencarian GWO
├── data/                       # Folder dataset lokal (diabaikan oleh git)
│   └── cell_images/
└── notebooks/
    └── malaria_classification_vgg19_gwo.ipynb   # Notebook eksperimen utama