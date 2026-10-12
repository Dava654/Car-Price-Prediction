# Penjelasan Kode & Tujuan Proyek (Car Dataset)

Dokumen ini berisi penjelasan alur kerja, tujuan, dan rincian kode yang
digunakan dalam pengolahan data serta pemodelan Machine Learning pada
dataset harga mobil (`car_dataset_2000_rows.csv`).

------------------------------------------------------------------------

## Untuk Apa Kode Ini?

Kode ini digunakan untuk **Data Preprocessing, Exploratory Data Analysis
(EDA), dan Pemodelan Machine Learning**.

Tujuan utamanya adalah menyiapkan data mobil agar dapat digunakan untuk
membangun model yang memprediksi **harga mobil (`price`)** berdasarkan
karakteristik kendaraan, seperti tahun, ukuran mesin, jarak tempuh,
merek, model, jenis bahan bakar, transmisi, tipe mobil, dan sistem
penggerak.

------------------------------------------------------------------------

## Penjelasan Detail Alur dan Fungsi Kode

### 1. Import Library

-   **Fungsi Kode:** `import pandas as pd`, `import numpy as np`,
    `import seaborn as sns`, `import matplotlib.pyplot as plt`
-   **Penjelasan:**
    -   **Pandas:** Membaca dan mengolah data berbentuk tabel.
    -   **NumPy:** Mendukung operasi numerik.
    -   **Seaborn dan Matplotlib:** Membuat grafik untuk membantu
        memahami data.
    -   **Scikit-Learn:** Digunakan pada tahap preprocessing, pembagian
        data, pelatihan model, dan evaluasi.

------------------------------------------------------------------------

### 2. Memuat dan Memeriksa Data (*Exploratory Data Analysis*)

-   **Fungsi Kode:** `pd.read_csv()`, `df.head()`, `df.info()`,
    `df.describe(include="all")`
-   **Penjelasan:**
    -   Membaca dataset `car_dataset_2000_rows.csv`.
    -   `head()` menampilkan lima baris pertama untuk melihat contoh
        data.
    -   `info()` menampilkan jumlah data dan tipe data setiap kolom.
    -   `describe(include="all")` menampilkan ringkasan statistik untuk
        kolom numerik maupun kategorikal.

------------------------------------------------------------------------

### 3. Penanganan Data Hilang (*Missing Values*)

-   **Fungsi Kode:** `df.isnull().sum()`, `fillna()`
-   **Penjelasan:**
    -   Memeriksa jumlah nilai kosong pada setiap kolom.
    -   **Kolom numerik:** Nilai kosong diisi menggunakan median kolom.
    -   **Kolom kategorikal:** Nilai kosong diisi dengan teks `Unknown`.
    -   Pemeriksaan dilakukan kembali setelah proses pengisian untuk
        melihat apakah masih ada nilai kosong.

------------------------------------------------------------------------

### 4. Penanganan Data Duplikat

-   **Fungsi Kode:** `df.duplicated().sum()`, `df.drop_duplicates()`
-   **Penjelasan:**
    -   Menghitung jumlah baris yang terduplikasi.
    -   Menghapus baris duplikat agar baris yang sama tidak tercatat
        berulang dalam dataset.

------------------------------------------------------------------------

### 5. Feature Engineering

-   **Fungsi Kode:** `fe = df.copy()`
-   **Penjelasan:**
    -   Membuat salinan data untuk menyiapkan fitur yang akan digunakan.
    -   Fitur numerik yang digunakan meliputi `year`, `engine_size`, dan
        `mileage`.
    -   Fitur kategorikal meliputi `brand`, `model`, `fuel_type`,
        `transmission`, `car_type`, dan `drive_type`.
    -   Kolom `price` digunakan sebagai target yang ingin diprediksi.

------------------------------------------------------------------------

### 6. Deteksi Pencilan (*Outliers*)

-   **Fungsi Kode:** `sns.boxplot()`, perhitungan **IQR (Interquartile
    Range)**

-   **Penjelasan:**

    -   Boxplot digunakan untuk melihat sebaran data dan nilai yang
        berpotensi menjadi pencilan.
    -   IQR dihitung menggunakan rumus:

    $$\text{IQR} = Q_3 - Q_1$$

    -   Batas pencilan diperiksa menggunakan rentang `Q1 - 1.5 × IQR`
        sampai `Q3 + 1.5 × IQR`.
    -   Pada notebook ini, pencilan hanya **diperiksa dan dihitung**.
        Data pencilan tidak dihapus atau diubah.

------------------------------------------------------------------------

### 7. Mengubah Data Kategorikal (*One-Hot Encoding*)

-   **Fungsi Kode:** `pd.get_dummies()`
-   **Penjelasan:**
    -   Kolom kategorikal seperti merek, model, jenis bahan bakar,
        transmisi, tipe mobil, dan sistem penggerak diubah menjadi
        beberapa kolom numerik.
    -   Setiap kategori direpresentasikan menggunakan nilai `0` atau
        `1`.
    -   Kolom `price` dipisahkan sebagai target (`y`), sedangkan kolom
        lainnya digunakan sebagai fitur (`X`).

------------------------------------------------------------------------

### 8. Analisis dan Visualisasi Data

-   **Fungsi Kode:** `sns.histplot()`, `sns.countplot()`,
    `sns.heatmap()`
-   **Penjelasan:**
    -   Histogram digunakan untuk melihat distribusi harga, tahun,
        ukuran mesin, dan jarak tempuh.
    -   Count plot digunakan untuk melihat jumlah mobil berdasarkan
        jenis bahan bakar, transmisi, tipe mobil, dan sistem penggerak.
    -   Heatmap digunakan untuk melihat korelasi antara fitur numerik
        dan harga mobil.
    -   Visualisasi ini membantu memahami karakteristik dataset sebelum
        pelatihan model.

------------------------------------------------------------------------

### 9. Pembagian Data (*Train-Test Split*)

-   **Fungsi Kode:** `train_test_split()`
-   **Penjelasan:**
    -   Dataset dibagi menjadi data latih dan data uji.
    -   Sebanyak **80%** data digunakan untuk melatih model dan **20%**
        digunakan untuk menguji model.
    -   `random_state=42` digunakan agar pembagian data dapat diulang
        dengan hasil yang konsisten.

------------------------------------------------------------------------

### 10. Standardisasi Fitur (*Feature Scaling*)

-   **Fungsi Kode:** `StandardScaler().fit_transform()`,
    `StandardScaler().transform()`
-   **Penjelasan:**
    -   Standardisasi menyamakan skala fitur numerik agar memiliki
        rata-rata mendekati `0` dan standar deviasi mendekati `1`.
    -   Scaler di-*fit* hanya pada data latih, lalu digunakan untuk
        mentransformasi data latih dan data uji.
    -   Data yang sudah distandardisasi digunakan untuk model **Linear
        Regression** pada notebook ini.

------------------------------------------------------------------------

### 11. Pelatihan Model (*Modeling*)

-   **Fungsi Kode:** `LinearRegression()`, `DecisionTreeRegressor()`,
    `RandomForestRegressor()`, `GradientBoostingRegressor()`
-   **Penjelasan:**
    -   **Linear Regression:** Memprediksi harga dengan memodelkan
        hubungan linear antara fitur dan target.
    -   **Decision Tree Regressor:** Membuat aturan berbentuk pohon
        untuk memperkirakan harga.
    -   **Random Forest Regressor:** Menggabungkan prediksi dari banyak
        pohon keputusan.
    -   **Gradient Boosting Regressor:** Membangun model secara bertahap
        untuk memperbaiki kesalahan prediksi sebelumnya.
    -   Keempat model dilatih menggunakan data latih untuk kemudian
        dibandingkan performanya.

------------------------------------------------------------------------

### 12. Evaluasi Model

-   **Fungsi Kode:** `mean_absolute_error()`, `mean_squared_error()`,
    `r2_score()`
-   **Penjelasan:**
    -   **MAE (Mean Absolute Error):** Rata-rata selisih absolut antara
        harga aktual dan harga prediksi. Semakin kecil, semakin baik.
    -   **MSE (Mean Squared Error):** Rata-rata kuadrat kesalahan
        prediksi. Semakin kecil, semakin baik.
    -   **RMSE (Root Mean Squared Error):** Akar dari MSE, sehingga
        satuannya sama dengan target harga. Semakin kecil, semakin baik.
    -   **R² Score:** Menggambarkan seberapa baik model menjelaskan
        variasi target. Nilai yang lebih tinggi umumnya lebih baik.
    -   Hasil evaluasi digunakan untuk membandingkan kinerja keempat
        model.

------------------------------------------------------------------------

### 13. Fitur Terpenting (*Feature Importance*)

-   **Fungsi Kode:** `GBR.feature_importances_`
-   **Penjelasan:**
    -   Menampilkan sepuluh fitur yang paling berpengaruh menurut model
        Gradient Boosting Regressor.
    -   Grafik membantu melihat fitur mana yang paling banyak digunakan
        model dalam membuat prediksi.
    -   Feature importance menunjukkan tingkat kepentingan fitur dalam
        model, bukan bukti bahwa fitur tersebut menyebabkan perubahan
        harga.

------------------------------------------------------------------------

## Sumber Dataset

Dataset yang digunakan adalah **Car Dataset for Price Prediction (2000
Rows)** dari Kaggle:

https://www.kaggle.com/datasets/abdulwadood11220/car-dataset-for-price-prediction-2000-rows

## File Proyek

-   `Laporan_Machine_Learning_Dava.ipynb` --- notebook berisi kode,
    analisis, preprocessing, pelatihan, dan evaluasi model.
-   `car_dataset_2000_rows.csv` --- dataset yang digunakan dalam
    analisis.

## Cara Menjalankan

1.  Simpan file notebook dan CSV di folder yang sama.

2.  Pastikan nama file CSV sama dengan nama yang digunakan pada
    `pd.read_csv()`.

3.  Instal library yang dibutuhkan:

    ``` bash
    pip install pandas numpy matplotlib seaborn scikit-learn joblib
    ```

4.  Buka notebook melalui Jupyter Notebook atau Visual Studio Code
    dengan ekstensi Jupyter.

5.  Jalankan cell secara berurutan dari awal dan periksa output setiap
    tahap.

## Catatan

Penjelasan ini mengikuti tahapan yang terdapat pada notebook. Hasil
akhir model bergantung pada data dan proses yang dijalankan. Pada bagian
pencilan, notebook hanya mendeteksi pencilan dan tidak menghapusnya.
