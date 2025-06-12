# 🌟 Future Path – Sistem Rekomendasi Jurusan Kuliah Berbasis NLP

**Future Path** adalah aplikasi web interaktif yang dirancang untuk membantu siswa SMA menentukan jurusan kuliah yang sesuai dengan minat, bakat, dan rencana karir mereka. Aplikasi ini dikembangkan sebagai bagian dari tugas akhir **Capstone Project Coding Camp 2025** oleh tim **CC25-CF323**.

## 🚩 Latar Belakang

Berdasarkan data dari Irene Guntur, Educational Psychologist dari Integrity Development Flexibility, **87% mahasiswa di Indonesia merasa salah jurusan**. Hal ini disebabkan oleh:

- Pemilihan jurusan yang kurang matang
- Pengaruh teman sebaya
- Kurangnya pemahaman terhadap minat dan potensi diri

Future Path hadir sebagai solusi dengan pendekatan berbasis **Artificial Intelligence (AI)** untuk memberikan rekomendasi jurusan yang lebih personal dan akurat.

---

## 🔍 Fitur Utama

- 🔠 Input narasi minat & kepribadian siswa
- 🤖 Rekomendasi jurusan dengan **Natural Language Processing (NLP)**
- 🌐 Antarmuka web modern, cepat, dan responsif

---

## 🧠 Teknologi yang Digunakan

### 📈 Model NLP dengan TensorFlow

Model deep learning dikembangkan menggunakan:

- **Embedding Layer**: Mewakili kata dalam bentuk vektor
- **Bidirectional LSTM**: Menangkap konteks dua arah dalam teks
- **Dense + ReLU + Dropout**: Untuk klasifikasi dan mencegah overfitting
- **Output Layer (Softmax)**: Menghasilkan rekomendasi jurusan

Model dilatih dengan:

- **Loss**: `SparseCategoricalCrossentropy`
- **Optimizer**: `Adam`
- **Metric**: `Accuracy`

### 💻 Web Development Stack

- **Frontend**:

  - [Bootstrap 5.3](https://getbootstrap.com/)
  - [SASS](https://sass-lang.com/)
  - [Webpack](https://webpack.js.org/)
  - [Babel](https://babeljs.io/)

- **Build Tools & Plugin**:

  - `HTMLWebpackPlugin` – Mengelola HTML otomatis
  - `MiniCssExtractPlugin` – Ekstrak CSS ke file terpisah
  - `CopyWebpackPlugin` – Menyalin file statis

- **Dev Server**:

  - `http-server` untuk local testing & debugging

- **Backend**:
  - [Hapi.js](https://hapi.dev/) – Framework Node.js ringan untuk API

---

## 🚀 Cara Menjalankan Proyek

### 1. Clone Repository

```bash
git clone https://github.com/username/repo-future-path.git
cd repo-future-path
```
