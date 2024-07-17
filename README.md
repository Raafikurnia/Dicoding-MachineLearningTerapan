# Laporan Proyek Machine Learning - PROYEK 1 : PREDIKSI MEDICAL CHARGES
# Raafi Kurnia Desita
![What-Is-A-Medical-Claim-MBA-Medical-Has-The-Solution_MBA-1080x675](https://github.com/user-attachments/assets/0932470a-3b92-4252-8271-9764006c5a64)


## Domain Proyek
Biaya medis atau asuransi medis merupakan salah satu aspek yang penting dalam perencanaan finansial. Prediksi medical charges atau biaya medis yang akurat adalah kunci bagi perusahaan  untuk menetapkan premi yang wajar dan mengelola risiko dengan lebih efektif. Dengan memanfaatkan machine learning, model prediksi medical charges yang mampu meramalkan biaya medis berdasarkan data yang dimiliki perusahaan dapat dibuat untuk dapat membantu perusahaan memanajemen prediksi untuk pengeluaran financial tentang biaya medis.

### Mengapa masalah ini harus diselesaikan:
1. Untuk perencanaan keuangan yang lebih baik
2. Untuk pengelolaan risiko yang lebih baik
3. Untuk peningkatan layanan pelanggan

### Bagaimana masalah ini dapat diselesaikan:
Masalah ini dapar diselesaikan dengan cara:
1. Melakukan pengumpulan data
2. Melakukan pra-pemrosesan data
3. Melakukan EDA dataset
4. Melakukan pemilihan model machine learning
5. Melakukan implementasi dan monitoring pada model
   

## Business Understanding
### Problem Statements
Berdasarkan domain proyek yang sebelumnya telah dibahas, berikut merupakan batasan masalah pada proyek yang dikerjakan:
  - Bagaimana kita dapat membuat prediksi biaya medis suatu individu dengan     memanfaatkan data yang ada?
  - Bagaimana kita dapat menemukan jenis algoritma terbaik yang akan digunakan dalam model prediksi biaya medis ini

### Goals
Berikut merupakan tujuan dari proyek yang dikerjakan:
- Membangun model machine learning yang dapat memberikan prediksi baya medis dengan akurat
- Melakukan evaluasi dan pembandingan kinerja berbagai model yang dibangun

### Solution statements
Berikut merupakan Solution statements dari proyek yang dikerjakan:
- Menggunakan 3 jenis algoritma berbeda untuk membangun model prediksi biaya medis. Ketiga algoritma tersebut berupa K-Nearest Neighbors (KNN), Random Forest (RF), dan Boosting. Lalu, melakukan evaluasi dan pembandingan terhadap ketiga algoritma untuk bisa mendapatkan model dengan prediksi yang paling efektif.
- Melakukan penyetelan hyperparameter tuning pada algoritma yang memiliki hasil evaluasi terbaik.


## Data Understanding
Dataset yang digunakan dalam proyek merupakan data open source yang dapat diakses melalui website Kaggle. Dataset Medical Cost dapat diakses <a href='https://www.kaggle.com/datasets/mirichoi0218/insurance'>disini</a>.

Pada dataset terdapat 1338 baris data yang masing-masing memiliki 7 kolom informasi. Kolom-kolom tersebut meliputi kolom age, sex, bmi, children, smoker, region dan charges dengan keterangan sebagai berikut:
![image](https://github.com/user-attachments/assets/f35c5aaf-ae51-4ed4-8a3e-d2112633f99e)
1.  age: usia individu (dalam tahun)
2.  sex: jenis kelamin individu (male/female)
3.  bmi: body mass index, merupakan  indikator pengukuran yang digunakan untuk menentukan kategori berat badan ideal atau tidak, diperoleh dari berat badan dibagi dengan kuadrat tinggi badan dalam satuan kg/m^2
4.  children: jumlah anak yang diasuransikan
5.  smoker: apakah individu merokok atau tidak (yes/no)
6.  region: kota tempat individu tinggal
7.  charges: biaya asuransi kesehatan yang dibebankan kepada individu

![image](https://github.com/user-attachments/assets/7287cb9f-070b-4433-b394-b241a3751265)

***numerical features***
![Untitled](https://github.com/user-attachments/assets/e0ff6a5b-3758-4e93-89df-413d31d2054e)

***Multivariate Analysis***
![Untitled](https://github.com/user-attachments/assets/3dac6340-b15a-480c-99c8-0143b4ee7ae4)


## Data Preparation
Teknik data preparation meliputi:
1. Pembersihan data. Pembersihan data yang dilakukan meliputi proses penanganan nilai null, penghapusan duplikat, dan penanganan outlier. 
penanganan nilai null:

![image](https://github.com/user-attachments/assets/17cfcf2c-70eb-42b9-941c-c9b1f339900b)

outlier pada bmi: ![image](https://github.com/user-attachments/assets/2d8fe214-62a9-469c-8aa2-df9833a36654)
outlier pada charges: ![image](https://github.com/user-attachments/assets/8aa9cd80-5c7c-448f-a9d5-93f0535ddcf1)

penanganan outlier:
![image](https://github.com/user-attachments/assets/d4faff33-9ac2-4f02-a7c9-edf55af22da0)

2. Encoding kategorical. Proses ini dilakukan dengan one-hot encoding atau pengubahan fitur kategorical menjadi representasi numerik. hal ini dilakukan sebab machine learning biasanya akan memerlukan data berbentuk numerik sebagai inputnya. Penerapan encoding kategorical dilakukan dengan menggunakan 'pandas.get_dummies()'
![image](https://github.com/user-attachments/assets/928481b2-6e70-4fbe-aa45-7d0a9c64c395)

3. Reduksi dimensi dengan Principal Component Analysis (PCA). proses ini bertujuan  untuk mengurangi jumlah fitur atau variabel dalam dataset sambil menjaga sebanyak mungkin informasi relevan dari data asli.
![image](https://github.com/user-attachments/assets/a02dd766-b427-43fc-9ee0-625f65fa48fe)
  
4. Pembagian data menjadi training dan testing. pembagian data atau split data dilakukan dengan perbandingan 90% untuk data training dan 10% untuk data testing. penerapannya dilakuakan dengan 'sklearn.model_selection.train_test_split()'
![image](https://github.com/user-attachments/assets/24753a70-a495-4617-973f-71f656b9939a)

5. Proses standarisasi. Proses ini merupakan proses penting dalam preparation data yang mengubah fitur numerik ke skala yang konsisten sehingga memiliki rata-rata 0 dan deviasi standar 1. proses ini dilakukan dengan tujuan untuk menghindari bias terhadap skala fitur, mempercepat konvergensi algoritma gradient descent dan meningkatkan interpretabilitas model.
![image](https://github.com/user-attachments/assets/a45682ed-5ed7-4fb9-a77c-d5c702471ff1)


## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.






## DOMAIN PROYEK
Menuliskan latar belakang yang relevan dengan proyek yang diangkat.
Jelaskan mengapa dan bagaimana masalah tersebut harus diselesaikan
Menyertakan hasil riset terkait atau referensi. Referensi yang diberikan harus berasal dari sumber yang kredibel dan author yang jelas

## BUSINESS UNDERSTANDING
Problem Statements (pernyataan masalah)
Goals (tujuan)
Menambahkan bagian “Solution Statement” yang menguraikan cara untuk meraih goals. Bagian ini dibuat dengan ketentuan sebagai berikut: 
Mengajukan 2 atau lebih solution statement. Misalnya, menggunakan dua atau lebih algoritma untuk mencapai solusi yang diinginkan atau melakukan improvement pada baseline model dengan hyperparameter tuning. Solusi yang diberikan harus dapat terukur dengan metrik evaluasi.

## Data Understanding
Memberikan informasi seperti jumlah data, kondisi data, dan informasi mengenai data yang digunakan 
Menuliskan tautan sumber data (link download).
Menguraikan seluruh variabel atau fitur pada data.
Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data atau exploratory data analysis.

## Data Preparation
Menerapkan dan menyebutkan teknik data preparation yang dilakukan.
Teknik yang digunakan pada notebook dan laporan harus berurutan.
Menjelaskan proses data preparation yang dilakukan
Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Membuat model machine learning untuk menyelesaikan permasalahan.
Menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.
Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. Jelaskan proses improvement yang dilakukan.
Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. Jelaskan mengapa memilih model tersebut sebagai model terbaik.

## Evaluation
Menyebutkan metrik evaluasi yang digunakan.
Menjelaskan hasil proyek berdasarkan metrik evaluasi.
Metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.
Menjelaskan metrik evaluasi yang digunakan untuk mengukur kinerja model. Misalnya, menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

