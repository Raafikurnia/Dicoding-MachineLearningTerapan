# Laporan Proyek Machine Learning - PROYEK 1 : PREDIKSI MEDICAL CHARGES
# Raafi Kurnia Desita
![What-Is-A-Medical-Claim-MBA-Medical-Has-The-Solution_MBA-1080x675](https://github.com/user-attachments/assets/0932470a-3b92-4252-8271-9764006c5a64)


## Domain Proyek
Biaya medis atau asuransi medis merupakan salah satu aspek yang penting dalam perencanaan finansial. Prediksi medical charges atau biaya medis yang akurat adalah kunci bagi perusahaan  untuk menetapkan premi yang wajar dan mengelola risiko dengan lebih efektif. Dengan memanfaatkan machine learning, model prediksi medical charges yang mampu meramalkan biaya medis berdasarkan data yang dimiliki perusahaan dapat dibuat untuk dapat membantu perusahaan memanajemen prediksi untuk pengeluaran financial tentang biaya medis.

**Rubrik/Kriteria Tambahan (Opsional)**:
- Jelaskan mengapa dan bagaimana masalah tersebut harus diselesaikan
- Menyertakan hasil riset terkait atau referensi. Referensi yang diberikan harus berasal dari sumber yang kredibel dan author yang jelas.


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
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

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

