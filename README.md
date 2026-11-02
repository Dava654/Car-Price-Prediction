# Penjelasan Kode & Tujuan Proyek (Car Dataset)

Halo, saya Mochamad Dava Dinata, mahasiswa Teknik Informatika Universitas Pamulang. Selamat datang di repositori tugas praktikum machine learning saya! 👋

Di repositori ini saya membangun model untuk memprediksi harga mobil (`price`) berdasarkan karakteristik kendaraannya, seperti merek, model, tahun, ukuran mesin, jarak tempuh, jenis bahan bakar, transmisi, tipe mobil, dan sistem penggerak. Seluruh prosesnya saya kerjakan dari nol sampai model jadi: mulai dari memuat dan memeriksa data, membersihkan, menganalisis, membagi data, melatih model, sampai mengevaluasi hasilnya pada dataset `car_dataset_2000_rows.csv`.

README ini saya tulis supaya siapa pun yang membuka repositori ini, bahkan yang belum pernah belajar machine learning, bisa paham apa yang saya kerjakan dan kenapa setiap langkahnya perlu dilakukan.

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

### 11. Penentuan Algoritma Regresi dan Pelatihan Model (*Modeling*)

-   **Fungsi Kode:** `LinearRegression()`, `DecisionTreeRegressor(max_depth=6)`,
    `RandomForestRegressor(n_estimators=200)`, `GradientBoostingRegressor()`
-   **Mengapa regresi?** Target `price` bertipe numerik kontinu (Rp 801.285
    -- Rp 7.997.116), jadi masalahnya regresi, bukan klasifikasi.
-   **Mengapa keempat algoritma ini?**

| Algoritma | Peran | Alasan pemilihan |
|---|---|---|
| Linear Regression | *Baseline* | Sederhana, mudah diinterpretasi; dipakai pada data hasil standardisasi |
| Decision Tree Regressor | Model non-linear dasar | Menangkap aturan bercabang antar fitur; `max_depth=6` membatasi overfitting |
| Random Forest Regressor | Ensemble *bagging* | Rata-rata banyak pohon, lebih stabil dan tahan noise |
| Gradient Boosting Regressor | Ensemble *boosting* | Memperbaiki kesalahan secara bertahap; pada referensi sering jadi yang terbaik |

-   **Catatan:** model tidak dipilih berdasarkan "algoritma mana yang pasti
    menghasilkan R² mendekati 1". Algoritma hanya **kandidat**; model terbaik
    ditentukan dari hasil evaluasi pada **data uji** (bagian 12 dan 14).
-   Linear Regression memakai data standardisasi; tiga model lainnya memakai
    data asli (model berbasis pohon tidak memerlukan scaling).

------------------------------------------------------------------------

### 12. Evaluasi Model dan Makna R²

-   **Fungsi Kode:** `mean_absolute_error()`, `mean_squared_error()`,
    `r2_score()`
-   **Metrik:**
    -   **MAE:** rata-rata selisih absolut harga aktual dan prediksi (makin kecil makin baik).
    -   **MSE / RMSE:** kuadrat / akar kuadrat kesalahan; RMSE satuannya sama dengan harga.
    -   **R² (koefisien determinasi):**

    $$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$

-   **Apakah R² harus mendekati 1?**
    -   R² = 1 berarti prediksi sempurna; R² = 0 berarti model sama saja
        dengan selalu menebak rata-rata harga; **R² < 0 berarti lebih buruk
        dari menebak rata-rata**.
    -   Tidak ada batas universal. Pada data pasar yang nyata, R² tinggi
        (mis. > 0,7) umumnya diharapkan, tetapi nilai itu hanya bermakna jika
        dihitung pada **data uji**, bukan data latih.
    -   R² yang mendekati 1 di data latih namun rendah di data uji adalah
        tanda *overfitting*, bukan model bagus.

------------------------------------------------------------------------

### 13. Fitur Terpenting (*Feature Importance*)

-   **Fungsi Kode:** `GBR.feature_importances_`
-   **Penjelasan:**
    -   Menampilkan sepuluh fitur yang paling berpengaruh menurut model
        Gradient Boosting Regressor.
    -   Feature importance menunjukkan tingkat kepentingan fitur dalam
        model, bukan bukti sebab-akibat terhadap harga.
    -   Karena performa model rendah (bagian 14), nilai importance di sini
        sebaiknya dibaca hati-hati dan tidak dijadikan kesimpulan kuat.

------------------------------------------------------------------------

### 14. Pembuktian Nilai R²: Benar atau Tidak?

Klaim yang diuji: *"model regresi pada dataset ini bisa mencapai R² mendekati
1 seperti pada referensi."* Pengujian dijalankan ulang dengan kode dan split
yang sama dengan notebook (`test_size=0.2`, `random_state=42`).

**a. Hasil pada data uji (400 baris)**

| Model | MAE | RMSE | R² uji | R² latih |
|---|---:|---:|---:|---:|
| Baseline (selalu tebak rata-rata) | 1.796.076 | 2.071.979 | -0,0005 | 0,0000 |
| Linear Regression | 1.818.910 | 2.108.046 | -0,0357 | 0,0194 |
| Decision Tree | 1.886.587 | 2.200.895 | -0,1289 | 0,1211 |
| Random Forest | 1.872.102 | 2.160.576 | -0,0879 | 0,8532 |
| Gradient Boosting | 1.840.285 | 2.134.893 | -0,0622 | 0,2169 |

**b. Validasi silang 5-fold (R² rata-rata)**: Linear Regression -0,031,
Random Forest -0,088, Gradient Boosting -0,054. Hasilnya konsisten dengan
split tunggal, jadi bukan kebetulan pembagian data.

**c. Mengapa R² rendah? Korelasi dan rata-rata harga**

-   Korelasi dengan `price`: `year` = 0,061; `engine_size` = 0,005;
    `mileage` = -0,031 (semuanya hampir nol).
-   Rata-rata harga per kategori hampir sama: antar `brand` hanya
    Rp 4,27 jt -- 4,41 jt; antar `fuel_type` Rp 4,22 jt -- 4,44 jt. Artinya
    fitur hampir tidak membedakan harga.
-   Contoh pola yang tidak masuk akal secara pasar: tiga Suzuki Swift tahun
    2008 berharga Rp 7,68 jt, Rp 6,82 jt, dan Rp 1,25 jt. Model dan tahun
    sama, tetapi harga berbeda sampai enam kali lipat. Ini mengindikasikan
    dataset kemungkinan **sintetis/acak** (dugaan berdasarkan pola di atas,
    bukan fakta dari sumber dataset).

**d. Kesimpulan**

-   **Klaim "R² mendekati 1" TIDAK terbukti pada dataset ini.** Semua R² uji
    bernilai negatif, artinya tidak ada model yang lebih baik dari sekadar
    menebak rata-rata harga.
-   Random Forest menunjukkan *overfitting* jelas: R² latih 0,853 tetapi R²
    uji -0,088.
-   Secara angka, di antara empat model Linear Regression paling baik (MAE,
    RMSE, R² paling kecil kerugiannya), tetapi **tetap tidak layak dipakai
    memprediksi harga**. Perbedaan antar model tidak bermakna karena semuanya
    mendekati tebakan rata-rata.
-   R² dalam referensi yang tinggi berasal dari dataset berbeda (data pasar
    nyata dengan hubungan fitur-harga yang kuat), sehingga tidak bisa
    disalin ke dataset ini. R² tidak "dipaksa" mendekati 1 dengan mengubah
    metrik; solusi yang jujur adalah mengganti/menambah data yang
    memang berhubungan dengan harga.

------------------------------------------------------------------------

## Referensi (Jurnal Nasional, 2022 -- 2023)

Hasil penelusuran hanya menemukan **satu** jurnal yang terverifikasi terbit
pada rentang 2022 -- 2023 dan sesuai topik:

1.  Hasibuan, E. & Karim, A. (2022). *Implementasi Machine Learning untuk
    Prediksi Harga Mobil Bekas dengan Algoritma Regresi Linear Berbasis Web.*
    **Jurnal Ilmiah KOMPUTASI**, 21(4), 595--602, Desember 2022.
    DOI: 10.32409/jikstik.21.4.3327
    https://ejournal.jak-stik.ac.id/index.php/komputasi/article/view/3327
    -   Isi (sudah dicek langsung pada PDF): regresi linear berganda dengan
        data Kaggle harga mobil bekas (10.668 baris, 9 kolom), pembagian
        data 80%:20%, dan evaluasi dengan R² serta RMSE.
    -   Hasil: **R² = 0,7622** (disebut "akurasi 76,22%") dan RMSE = 3.813,65.
    -   Makna bagi proyek ini: pada data dengan hubungan fitur-harga yang
        nyata, R² ≈ 0,76 sudah dianggap layak oleh penulis; jadi R² tidak
        harus tepat 1. Nilai ini dihitung pada dataset yang berbeda,
        sehingga tidak bisa langsung dibandingkan dengan hasil di atas.

Referensi pendukung (ditemukan, tetapi **di luar rentang tahun 2022--2023**,
jangan dipakai sebagai referensi utama sesuai syarat):

-   Kriswantara, B., Kurniawati, & Pardede, H. F. (2021). *Prediksi Harga Mobil
    Bekas dengan Machine Learning.* Syntax Literate, 6(5). (R² = 0,88 DNN;
    0,82 Random Forest)
-   *Evaluasi Model Machine Learning untuk Prediksi Harga Mobil dengan
    Perbandingan Ensemble dan Regresi Linear.* JIRSI (Unity Academy).
    (R² Gradient Boosting 0,9639; Random Forest 0,8997; Linear Regression 0,4179)

Catatan: seluruh R² tinggi pada referensi dihitung pada dataset yang berbeda,
sehingga hanya menjadi pembanding konsep, bukan target yang pasti tercapai di
dataset ini.

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

Hal yang tidak sinkron di notebook dan perlu dirapikan (README ini hanya
mendokumentasikan, tidak mengubah notebook):

-   Teks pengantar di notebook menyebut LARS, tetapi LARS tidak
    diimplementasikan; `LogisticRegression` juga di-*import* namun tidak
    dipakai. Kode hanya memakai empat model regresi di atas.
-   Cell terakhir menyimpan `GradientBoostingClassifier` sebagai
    `best_model`, padahal target `price` adalah regresi. Sebaiknya yang
    disimpan `GBR` (atau model terpilih), dan `import joblib, pickle` perlu
    ditambahkan di cell tersebut.
