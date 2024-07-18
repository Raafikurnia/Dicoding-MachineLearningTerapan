# Laporan Proyek Machine Learning - PROYEK 1 : PREDIKSI MEDICAL CHARGES
### Nama: Raafi Kurnia
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
  - Bagaimana kita dapat menemukan jenis algoritma terbaik yang akan digunakan dalam model prediksi biaya medis ini?

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
Model machine learning yang digunakan untuk menyelesaikan permasalahan ini meliputi model machine learning dengan algoritma K-Nearest Neighbors (KNN), Random Forest (RF) dan Boosting.
1. Algoritma K-Nearest Neighbors (KNN)
   -Kelebihan: sederhana dan mudah diimplementasikan serta dapat bekerja dengan baik pada dataset berskala kecil
   -Kekurangan: kinerja dapat memburuk pada dataet besar, membutuhkan lebih banyak waktu
   
2. Algoritma Random Forest (RF)
   -Kelebihan: mengurangi risiko overfitting pada model, dapat menangani data non-linier dan interaksi fitur dengan baik
   -Kekurangan: model lebih kompleks, memerlukan lebih banyak waktu
   
3. Algoritma Boosting
   -Kelebihan: kinerja yang baik dalam berbagai kasus, dapat menangani data non-linier dan interaksi fitur dengan baik
   -Kekurangan : Sensitif terhada noise, memerlukan tuning hyperparameter yang cermat


### Cara kerja masing-masing algoritma:
1. Algoritma KNN

   ```
   knn = KNeighborsRegressor(n_neighbors=10)
   knn.fit(X_train, y_train)
   
   models.loc['train_mse','knn'] = mean_squared_error(y_pred =       knn.predict(X_train), y_true=y_train)
   ```
   Kode ini melatih model K-Nearest Neighbors (KNN) dengan 10 tetangga pada    data pelatihan X_train dan y_train. Setelah pelatihan, model ini digunakan untuk memprediksi nilai pada data pelatihan, dan MSE (Mean Squared Error) dari prediksi tersebut dihitung dan disimpan dalam DataFrame models pada baris 'train_mse' dan kolom 'KNN'

   

2. Algoritma RF
   ```
   RF = RandomForestRegressor(n_estimators=50, max_depth=16, random_state=55, n_jobs=-1)
   RF.fit(X_train, y_train)
   models.loc['train_mse','RandomForest'] = mean_squared_error(y_pred=RF.predict(X_train), y_true=y_train)
   ```

   Kode ini melatih model RandomForestRegressor dengan 50 estimator dan kedalaman maksimum 16 pada data pelatihan X_train dan y_train. Setelah pelatihan, model digunakan untuk memprediksi nilai pada data pelatihan, dan MSE (Mean Squared Error) dari prediksi tersebut dihitung serta disimpan dalam DataFrame models pada baris 'train_mse' dan kolom 'RandomForest'



3. Algoritma Boosting

   ```
   boosting = AdaBoostRegressor(learning_rate=0.05, random_state=55)
   boosting.fit(X_train, y_train)
   models.loc['train_mse','Boosting'] = mean_squared_error(y_pred=boosting.predict(X_train), y_true=y_train)
   ```
   Kode ini melatih model AdaBoostRegressor dengan laju pembelajaran 0.05 pada data pelatihan X_train dan y_train. Setelah pelatihan, model digunakan untuk memprediksi nilai pada data pelatihan, dan MSE (Mean Squared Error) dari prediksi tersebut dihitung serta disimpan dalam DataFrame models pada baris 'train_mse' dan kolom 'Boosting'.

   

## Evaluation
Pada proyek ini, metrik evaluasi yang diggunakan berupa MSE atau Mean Squared Error. MSE merupakan metrik evaluasi yang mengukur rata-rata kuadrat kesalahan antara nilai prediksi dan nilai aktual.

### Hasil Perhitungan MSE pada masing-masing algoritma
![image](https://github.com/user-attachments/assets/b7eb3323-6f41-4e18-beab-e5d6751da581)
![Untitled](https://github.com/user-attachments/assets/373a498a-bd24-4b73-b97f-3b810023546d)


### Hasil Perbandidngan Prediksi menggunakan 3 jenis algoritma
![image](https://github.com/user-attachments/assets/e79dfa29-91f9-4614-8595-b57c00368e32)

### Kesimpulan
Berdasarkan hasil prediksi tersebut, RF memberikan prediksi yang paling mendekati nilai sebenarnya (y_true). Sehingga model RF dapat digunakan sebagai model terbaik untuk melakukan prediksi medical charges.

Dalam hal ini, berati kita telah menjawab pertanyaan pada problem statement dan mencapai tujuan dari goals yang sebelumnya telah disebutkan.
Menjawab peoblem statement:
  - Bagaimana kita dapat membuat prediksi biaya medis suatu individu dengan     memanfaatkan data yang ada?
= Kita dapat membuat prediksi biaya medis dengan menggunakan model yang telah dilatih untuk memprediksi biaya berdasarkan fitur yang terdapat dalam data  individu.
  - Bagaimana kita dapat menemukan jenis algoritma terbaik yang akan digunakan dalam model prediksi biaya medis ini?
= Kita dapat menemukan jenis algoritma terbaik yang akan digunakan dalam model prediksi dengan membandingkan performa beberapa algoritma, dari hal tersebut kita telah  menemukan bahwa algoritma RandomForestRegressor adalah yang paling efektif dalam menghasilkan prediksi yang akurat untuk biaya medis.

Berdasarkan hasil analisis dan evaluasi yang dilakukan, kita berhasil mencapai tujuan proyek yang diharapkan yaitu:
    -Berhasil membangun model yang akurat untuk memprediksi biaya medis melalui model RandomForestRegressor yang telah menunjukkan performa terbaik dengan prediksi yang paling mendekati nilai sebenarnya (y_true).
   -Berhasil melakukan evaluasi dan perbandingan kinerja berbagai model yang dibangun, yaitu dengan membandingkan Kinerja berbagai model (KNN, RandomForest, dan Boosting) dan memberikan hasil berupa kinerja RandomForestRegressor  menjadi model yang paling efektif berdasarkan MSE pada data pengujian.
Dengan demikian, tujuan proyek untuk membangun model prediksi yang akurat dan mengevaluasi kinerja berbagai model telah berhasil dicapai.
