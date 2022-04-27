# Laporan Proyek Machine Learning - Subkhan Rian Romadhon

#### **Contents:**
  - [Project Domain](#project-domain)
  - [Business Understanding](#business-understanding)
    - [Problem Statement](#1-problem-statement)
    - [Goals](#2-goals)
    - [Solutions](#3-solutions)
  - [Data Understanding](#data-understanding)
    - [Anime](#1-anime)
      - [Menamplikan 5 Data Teratas](#11-menamplikan-5-data-teratas)
      - [Mengecek Tipe Data dan Missing Value ](#12-mengecek-tipe-data-dan-missing-value)
      - [Statistika Deskriptif dan Distribusi Genre](#13-statistika-deskriptif-dan-distribusi-genre)
     - [Rating](#2-rating)
        - [Menamplikan 5 Data Teratas](#21-menampilkan-5-data-teratas)
        - [Mengecek Tipe Data dan Missing Value ](#22-mengecek-tipe-data-dan-missing-value)
        - [Statistik Rating](#23-statistik-rating)
  - [Data Preparation](#data-preparation)
      - [Penyesuaian Tipe Data](#1-penyesuaian-tipe-data)
      - [Menghapus Baris dengan Missing Value dan Noise](#2-menghapus-baris-dengan-missing-value-dan-noise)
      - [Perbaikan Penulisan Jenis Genre](#3-perbaikan-penulisan-jenis-genre)
      - [Pemilihan Atribut](#4-pemilihan-atribut)
      - [Melakukan Encoding Atribut user_id dan anime_id](#5-melakukan-encoding-atribut-user_id-dan-anime_id-collaborative-filtering)
      - [Membagi Dataser menjadi Train dan Validation](#6-membagi-dataset-menjadi-train-dan-validation-collaborative-filtering)
  - [Modeling and Result](#modeling-and-result)
    - [Content Based Filtering](#1-content-based-filtering)
    - [Collaborative Filtering](#2-collaborative-filtering)
  - [Evaluasi](#evaluasi)
    - [Content Based Filtering](#1-content-based-filtering-1)
    - [Collaborative Filtering](#2-collaborative-filtering-1)
  - [Daftar Pustaka](#daftar-pustaka)

## Project Domain

<img src="https://image.myanimelist.net/ui/5LYzTBVoS196gvYvw3zjwLgAvBFCaQ-G87EPlpMcd2s" width=400/>

Dewasa ini industri hiburan semakin berkembang, salah satunya ialah Anime. Anime diambil dari kata "animation" dalam bahasa Inggris merupakan animasi yang diproduksi di Jepang menggunakan teknologi komputer maupun manual dengan tangan. Anime mulai populer di kalangan masyarakat semenjak kehadiran Astro Boy karya Ozamu Tezuka pada tahun 1963. Hingga saat ini, anime sudah sangat berkembang dari segi grafis, musik maupun alur cerita yang lebih menarik. Sebuah data dari situs epicdope.com menampilkan 10 negara dengan jumlah peminat anime di tahun 2022. Negara di luar Jepang, yakni Amerika menempati posisi ke 2 dimana sebanyak 71.86% dari total penduduk telah menonton anime. 

<img src="https://user-images.githubusercontent.com/61647791/164499658-3477255a-911e-4714-9253-4403113714a1.png" width=600 />

Anime memiliki karakteristik yang umumnya bisa dilihat dari segi genre. Terdapat berbagai genre anime seperti horror, thriller, comedy, adventure dan sebagainya. Sebuah masalah muncul ketika pengguna ingin melihat anime dengan karakteristik sesuai dengan yang diinginkan, misalnya dari segi genre, rating, maupun tipe anime (movie, TV, OVA).  Oleh karena itu, muncul suatu ide untuk menerapkan sistem rekomendasi untuk memilih anime yang tepat sesuai keinginan pengguna. Terdapat beberapa manfaat apabila perusahaan streaming anime seperti <a href="https://www.crunchyroll.com/">Crunchyroll</a> menerapkan sistem rekomendasi, misalnya dapat meningkatkan jumlah anime yang ditonton, menyediakan anime yang beragam, meningkatkan kepuasan pengguna, serta bisa memahami preferensi pengguna terkait anime yang diinginkan.

## Business Understanding
Berdasarkan pemaparan di atas, berikut merupakan permasalahan beserta tujuan dibuatnya sistem rekomendasi anime:

### 1. Problem Statement
- Bagaimana cara membuat sistem rekomendasi anime yang merekomendasikan pengguna berdasarkan genre anime?
- Dengan menggunakan data rating yang dimiliki pengguna, bagaimana perusahaan jasa streaming dapat merekomendasikan anime yang belum pernah ditonton pengguna?

### 2. Goals
Untuk menjawab permasalahan tersebut dibuatlah sistem rekomendasi dengan tujuan sebagai berikut:
- Menghasilkan rekomendasi anime sebanyak N buah kepada pengguna berdasarkan genre.
- Menghasilkan beberapa rekomendasi anime yang sesuai dengan preferensi pengguna dan belum pernah ditonton.

### 3. Solutions
Untuk merealisasikan tujuan di atas dibuatlah dua jenis sistem sistem rekomendasi, yakni content based filtering dan collaborative filtering. Content based merupakan sistem akan merekomendasikan anime kepada pengguna berdasarkan karakteristik yang dimiliki anime, misalnya dari genre, penulis, maupun tipe anime (movie, TV, OVA). Di sisi lain, collaborative filtering merupakan sistem yang merekomendasikan anime yang belum pernah ditonton menggunakan data historis anime yang pernah ditonton sebelumnya. Dengan menggunakan kedua metode ini diharapkan dapat menghasilkan rekomendasi yang akurat sesuai dengan yang diinginkan pengguna. 

## Data Understanding

Dataset yang digunakan diambil dari API <a href="https://myanimelist.net/">myanimelist</a> dan terdiri dari 73,516 pengguna dan 12,294 anime. Dataset dapat diunduh melalui <a href="https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database">situs kaggle ini</a>. Terdapat dua buah file data berformat csv, yakni anime.csv dan rating. Keduanya akan dibahas terpisah beserta eksplorasi data untuk masing-masing file.

### 1. Anime
Dataset ini berisi beragam anime yang tersedia di situs myanimelist yang terdiri dari beberapa variabel, di antaranya:
- anime_id = ID unik dari myanimelist.net untuk mengidentifikasi anime.
- name = nama anime.
- genre  = list genre dari anime.
- type = jenis anime, apakah berupa movie, TV, OVA, dan sebagainya.
- episodes = jumlah episode yang dimiliki, apabila bertipe movie maka hanya memiliki 1 episode.
- rating = rata-rata rating anime dengan skala 1-10.
- members  = jumlah member komunitas anime tersebut. 

#### 1.1. Menamplikan 5 Data Teratas

<img src="https://user-images.githubusercontent.com/61647791/165239950-186de91d-a0a2-4d6d-a700-fbf255aa3c64.png">

Seketika saya melihat ada nama anime yang mirip seperti Gintama* dan Gintama&#039. Keduanya sama-sama memiliki 51 episode dengan rating dan member yang berbeda. Saya tidak akan melakukan penggabungan data karena anime_id nya berbeda sehingga diasumsikan keduanya merupakan anime yang berbeda seri. Selanjutnya kita akan melihat informasi tipe data masing-masing atribut dan mengecek apakah terdapat missing value pada file anime.csv.

#### 1.2. Mengecek Tipe Data dan Missing Value 
<img src="https://user-images.githubusercontent.com/61647791/165240229-dfa7ad78-2809-45f9-9178-e49b31581f9e.png">

Atribut anime_id seharusnya merupakan data nominal yang bersifat kategorikal, bukan data numerikal (int64). Untuk mencegah terjadinya operasi perhitungan pada atribut anime_id, maka pada tahap data preparation akan diubah tipe datanya menjadi object. Selain itu terdapat nilai null pada atribut genre, type dan rating. Dikarenakan kita akan membuat sistem rekomendasi berdasarkan genre, maka nilai yang kosong pada atribut genre akan kita hapus. Kita tidak akan menghapus nilai kosong pada atribut rating maupun type karena keduanya tidak diperlukan untuk membuat sistem rekomendasi berdasarkan genre. Untuk membuat sistem rekomendasi tersebut, kita hanya memerlukan atribut anime_id, name dan genre.

Selanjutnya kita akan melihat statistika deskriptif untuk kolom **rating** dan **members**. Kita akan melihat berapakah rata-rata member di seluruh anime tersebut, maupun rata-rata rating terendah dan tertinggi. Selain itu, kita juga akan melihat distribusi genre anime untuk melihat kira-kira genre apa yang paling mendominasi.

#### 1.3. Statistika Deskriptif dan Distribusi Genre

<img src="https://user-images.githubusercontent.com/61647791/165240851-bc771db5-e150-466c-919c-1ad58b168aef.png">

<img src="https://user-images.githubusercontent.com/61647791/165241005-771d161d-52fb-43bb-994b-5c7e7d3da344.png">

Dari statistika deskriptif kita bisa mendapatkan beberapa informasi, di antaranya:
- Rating terendah dan tertinggi seluruh anime berturut-turut sebesar 1.67 dan 10.
- Jumlah member komunitas anime terendah dan tertinggi berturut-turut sebesar 5 dan 1,013,917.
- Distribusi rating bentuknya simetris dengan rata-rata rating yang sering muncul sebesar +- 7.
- Comedy menjadi genre yang paling banyak ditemukan dalam koleksi anime tersebut, diikuti dengan action dan adventure. 
- Terdapat jenis genre yang terdiri dari 2 kata atau lebih, misalnya slice of life. Spasi pada kategori tersebut harus kita ganti dengan tanda garis bawah ( _ ) supaya pada saat menggunakan TfidfVectorizer kata-katanya tidak terpisah menjadi slice, of, dan life.

### 2. Rating
Dataset ini berisi penilaian pengguna pada sebuah anime dengan beberapa atribut sebagai berikut:
- user_id = ID yang dimiliki pengguna.
- anime_id = ID unik anime yang pernah ditonton pengguna.
- rating = rating yang diberikan pengguna, bernilai -1 jika pengguna hanya menonton saja dan tidak memberikan rating.

#### 2.1. Menampilkan 5 Data Teratas

<img src="https://user-images.githubusercontent.com/61647791/165242268-4f879ac4-0223-4cb5-b56f-67fae3ee08c9.png">

Sesuai dengan penjelasan dataset, terdapat rating yang bernilai -1, artinya pengguna tidak memberikan rating dan hanya menonton saja. Kita tidak tahu apakah pengguna tersebut benar-benar menyukai anime yang ditonton atau tidak sehingga hal ini bisa berpotensi menimbulkan bias pada model rekomendasi. Kita akan menghapus semua data dengan rating bernilai -1. Selanjutnya kita akan melihat informasi tipe data dan missing value pada masing-masing atribut pada file rating.csv.

#### 2.2. Mengecek Tipe Data dan Missing Value

<img src="https://user-images.githubusercontent.com/61647791/165242562-a0c5eba6-2d70-45b6-8114-9c16be1a7eb1.png">

Sama halnya dengan dataset anime, atribut anime_id juga masih bertipe integer pada dataset rating. Oleh karena itu, pada tahap data preparation atribut anime_id akan diubah tipenya menjadi object. Selain itu, tidak ada missing value pada semua atribut dataset rating.

#### 2.3. Statistik Rating

<img src="https://user-images.githubusercontent.com/61647791/165243037-dd09126a-8961-4153-a937-35fcb9ac6fdc.png">

Berdasarkan statistik deskriptif maupun histogram dapat dilihat bahwa rating yang bernilai -1 masih banyak muncul pada dataset rating. Selain itu, terdapat kecenderungan pengguna memberikan rating yang tinggi pada anime.

## Data Preparation

#### 1. Penyesuaian Tipe Data

Kita akan mengubah tipe data atribut anime_id dari yang semula bertipe integer menjadi objek karena anime_id merupakan data nominal dan untuk mengantisispasi terjadinya operasi perhitungan. Berikut merupakan output hasil normalisasi yang ditunjukkan dengan perubahan tipe data pada atribut anime_id baik pada dataset anime maupun rating:

<img src="https://user-images.githubusercontent.com/61647791/165243418-cc05794a-f257-4a4f-9a04-c6b6c139510f.png">
<br>

<img src="https://user-images.githubusercontent.com/61647791/165243515-11fda977-8146-4165-8ec3-e33d877e00f6.png">

#### 2. Menghapus Baris dengan Missing Value dan Noise
Pada dataset anime kita akan menghapus baris dengan missing value pada atribut genre. Hal ini karena sistem rekomendasi yang akan dibangun (content based) memerlukan atribut genre yang memiliki value. Di sisi lain, pada dataset rating terdapat rating yang bernilai -1 sehingga berpotensi menjadi noise ketika digunakan untuk membuat sistem rekomendasi berbasis collaborative filtering. 

<img src="https://user-images.githubusercontent.com/61647791/165243809-faaf2c84-9d22-467e-b228-209240408e6b.png">
<br>
<img src="https://user-images.githubusercontent.com/61647791/165243980-350f3258-f23a-44ef-9196-681b65473e7c.png">

#### 3. Perbaikan Penulisan Jenis Genre

Diketahui masih terdapat jenis genre yang terdiri dari 2 kata atau lebih, misalnya slice of life. Spasi pada kategori tersebut harus kita ganti dengan tanda garis bawah ( _ ) supaya pada saat menggunakan TfidfVectorizer kata-katanya tidak terpisah menjadi slice, of, dan life. Berikut merupakan tampilan 5 data teratas hasil perbaikan penulisan jenis genre. Atribut "genre_clean" merupakan hasil perbaikan dari atribut "genre".

<img src="https://user-images.githubusercontent.com/61647791/165244407-a2ef9f88-a4bb-47d8-a604-c5d916bc888e.png">

#### 4. Pemilihan Atribut
Untuk membangun sistem rekomendasi berbasis genre, atribut yang akan digunakan yakni anime_id, name, dan genre. Di sisi lain, untuk membangun sistem rekomendasi dengan metode collaborative filtering, atribut yang digunakan yakni user_id, anime_id, dan rating ditambah nama anime dan genre pada dataset anime. Oleh karena itu dilakukan penggabungan data rating dengan anime menggunakan inner join karena tidak semua anime diberikan rating oleh pengguna.

- Dataset untuk Content Based Filtering
  <img src="https://user-images.githubusercontent.com/61647791/165244725-11afa727-8c8c-4b2c-94e8-96f8fe59937b.png">

- Dataset untuk Collaborative Filtering
  <img src="https://user-images.githubusercontent.com/61647791/165244934-cd714a3c-4bf2-40d5-a189-521d27b593c2.png">

#### 5. Melakukan Encoding Atribut user_id dan anime_id (Collaborative Filtering)

Model collaborative filtering yang akan dibangun memanfaatkan model deep learning sehingga perlu dilakukan pemrosesan awal pada data seperti encoding atribut yang bersangkutan. Pada tahap ini dilakukan proses encoding atribut "user_id" dan "anime_id" ke dalam bilangan integer. Setelah itu, kedua atribut dipetakan ke dalam dataframe yang berkaitan yakni "col_fil". Terakhir adalah memastikan bahwa atribut rating memiliki tipe data float. Berhubung atribut ini masih berupa integer, maka dilakukan perubahan tipe data menjadi float. Berikut merupakan tampilan dataset rating yang sudah dilakukan encoding atribut "user_id" dan "anime_id". Perubahan dapat dilihat pada atribut "user" dan "anime".

<img src="https://user-images.githubusercontent.com/61647791/165272619-0cdd05e9-fb2a-4377-893c-5bb503b36d06.png">

##### 6. Membagi Dataset Menjadi Train dan Validation (Collaborative Filtering)
Setelah melakukan encoding, selanjutnya melakukan pembagian dataset dengan persentase 80% untuk pelatihan dan 20% untuk validasi. Supaya pembagian data terdistribusi secara random maka dilakukan pengacakan terlebih dahulu pada dataset rating. Variabel independent (x) yang akan digunakan yakni atribut user dan anime sedangkan variabel dependennya (y) adalah rating yang telah dinormalisasi supaya memudahkan dalam proses pemodelan. Metode normalisasi yang digunakan yakni min max scaler.

<img src="https://www.oreilly.com/library/view/regression-analysis-with/9781788627306/assets/ffb3ac78-fd6f-4340-aa92-cde8ae0322d6.png" width=300>

## Modeling and Result

### 1. Content Based Filtering

Content based filtering merupakan sistem rekomendasi personalized yang merekomendasikan item yang mirip dengan item yang disukai pengguna di masa lampau. Algoritma ini bekerja dengan memberikan saran berdasarkan kemiripan item yang direkomendasikan dengan item yang disukai dari data objek seperti genre anime, penulis, sutradara, artis dan lain-lain. Berikut merupakan ilustrasi sistem rekomendasi yang akan dibuat berdasarkan kemiripan genre.

<img src="https://user-images.githubusercontent.com/61647791/164959912-a113ff8b-4961-4c55-a377-3aa1ec196f40.png" width=500>

Menurut Lops dkk. (2010) sistem rekomendasi berbasis konten memiliki keunggulan serta kekurangan sebagai berikut:

**Keunggulan:**
- **User independence**: Sistem rekomendasi tidak bergantung pada user lainnya. Sistem membangun profil dengan cara mengeksploitasi penilaian pengguna aktif.
- **Transparency**: Cara kerja sistem rekomendasi dijelaskan dengan rinci dalam memunculkan item yang relevan berdasarkan fitur konten.
- **New Item:** Mampu merekomendasikan item yang belum pernah dinilai oleh pengguna.

**Kekurangan:**
- **Limited Content Analysis:** Memiliki keterbatasan dalam jumlah maupun jenis fitur yang terkait, begitupula dengan item-item yang disarankan.  
- **Over-Specialization:** Sistem tidak memiliki metode yang melekat untuk menemukan sesuatu yang tidak terduga. Sistem akan menunjukkan item yang nilainya tinggi, kemudian dicocokkan dengan profil pengguna, sehingga akan selalu menemukan item serupa seperti yang sudah direkomendasikan sebelumnya.
- **New User:** Sistem tidak dapat memberikan rekomendasi yang handal pada pengguna baru, dikarenakan membutuhkan penelusuran terkait preferensi pengguna.

Untuk membangun sistem rekomendasi berbasis konten, khususnya genre hal pertama yang harus dilakukan adalah membuat matriks korelasi anime dengan genre berukuran (mxn) di mana m menunjukkan nama anime, sedangkan n menunjukkan genre. Untuk membuat matriks tersebut kita akan menggunakan TfidfVectorizer dari library scikit-learn. Berikut merupakan bentuk dari matriks tf-idf yang diubah menjadi dataframe. Saya hanya mengambil sampel 10 anime dan 10 jenis genre karena matriks terlalu besar.

<img src="https://user-images.githubusercontent.com/61647791/165273641-54cd4db1-13b6-4d46-9991-c16793b0e3cf.png">

Dari pembuatan matriks korelasi anime dengan genre diperoleh matriks berukuran 12232x43. Pada tahap selanjutnya, dibuatlah matriks kesamaan antar anime berukuran (mxm). Semakin mirip anime satu dengan lainnya maka skor kesamaan akan mendekati satu, begitupun sebalinkya. Untuk menghitung nilai kesamaan antar anime digunakanlah cosine similarity. Cosine similarity akan menghitung sudut cosinus antara dua vektor. Semakin kecil sudut cosinus-nya maka semakin besar cosine similarity. Metrik ini seringkali digunakan mengukur kesamaan antar teks atau kata sehingga cocok digunakan dalam kasus ini. Berikut merupakan ilustrasi dari cosine similarity beserta hasil matriks kesamaan untuk 10 pasangan anime saja karena jumlahnya terlalu banyak.    
<img src="https://www.researchgate.net/publication/320914786/figure/fig2/AS:558221849841664@1510101868614/The-difference-between-Euclidean-distance-and-cosine-similarity.png" width=400>

<img src="https://i0.wp.com/clay-atlas.com/wp-content/uploads/2020/03/cosine-similarity-2.png?fit=800%2C208&ssl=1" width=400>

<img src="https://user-images.githubusercontent.com/61647791/165274117-838447a3-693b-4a54-92b9-038a337afae9.png">

Untuk mendapatkan top N rekomendasi berdasarkan genre, dibuatlah suatu fungsi dengan parameter nama anime, dataframe kesamaan berukuran nxn, kolom yang ditampilkan serta jumlah rekomendasi yang diinginkan. Kita akan menggunakan fungsi tersebut untuk memunculkan top 5 rekomendasi yang memiliki kemiripan dengan anime Fullmetal Alchemist. 

<img src="https://user-images.githubusercontent.com/61647791/165274649-6164dcd0-d695-4d9a-b118-1de90706bdca.png">

<img src="https://user-images.githubusercontent.com/61647791/165274831-e18b81ed-9b2e-412e-b322-01dae3618f50.png">

### 2. Collaborative Filtering

Kita akan membuat sistem rekomendasi collaborative filtering yang memanfaatkan penerapan deep learning. Sistem rekomendasi ini akan memanfaatkan data rating yang telah diberikan pengguna pada suatu anime kemudian melakukan prediksi anime yang disukai dan belum pernah dilihat. Hal ini bisa menjadi kekurangan apabila data rating anime yang diberikan user tidak tersedia. Kita akan menggunakan library keras dan tensorflow kemudian membuat kelas serta fungsi untuk membangun sistem rekomendasi berbasis deep learning. 

Pada proses pelatihan, model akan menghitung skor kecocokan antara pengguna dengan anime dengan rentang 0 sampai 1. Tahap pertama yakni melakukan proses embedding pada data user dan anime. Setelah itu, dilakukan operasi dot product antara embedding user dengan anime kemudian ditambahkan bias untuk setiap user dan anime. Fungsi aktivasi sigmoid digunakan untuk menghitung skor kecocokan berdasarkan hasil perhitungan tersebut.

<img src="https://user-images.githubusercontent.com/61647791/165275229-e9d9f21e-42c1-4bde-be6c-b562b2edd0bf.png">

Untuk menguji model rekomendasi ini, kita akan mengambil sampel user secara acak serta mendefinisikan variabel bernama **anime_not_watched** yang menunjukkan anime yang belum pernah ditonton oleh pengguna tersebut. Variabel **anime_not_watched** akan menjadi anime yang direkomendasikan. Berikut merupakan hasil rekomendasi menggunakan metode collaborative filtering.

<img src="https://user-images.githubusercontent.com/61647791/165275789-1de7c0bb-3c06-469f-9467-c95901bceedd.png">

## Evaluasi

#### 1. Content Based Filtering

Untuk mengukur performa model rekomendasi berbasi konten, digunakanlah metrik berupa presisi yang dihitung dengan cara membagi jumlah prediksi yang tepat terhadap seluruh seluruh prediksi yang salah maupun tepat. Berikut merupakan rumus dan ilustrasi dari metrik presisi, khususnya dalam kasus ini.

<img src="https://user-images.githubusercontent.com/61647791/165103309-26b85e6c-3fce-4851-9442-75d92c565d60.png" width=500>


Dikarenakan terdapat lebih dari satu genre dalam satu anime, skor dihitung berdasarkan rata-rata presisi kecocokan masing-masing genre dalam anime yang telah ditonton terhadap masing-masing genre pada anime yang direkomendasikan.

<img src="https://user-images.githubusercontent.com/61647791/165105550-a7e97667-4ea2-4ac0-83ee-bb87901c6099.png" width=500>

Berikut merupakan skor presisi sistem rekomendasi berbasis konten, dalam hal ini menggunakan anime Fullmetal Alchemist.

<img src="https://user-images.githubusercontent.com/61647791/165108457-99488a1d-6a1d-4d29-a71b-2bfd81e36ab3.png" >

#### 2. Collaborative Filtering

Performa model rekomendasi collaborative filtering yang menggunakan deep learning diukur menggunakan metrik root mean square error (RMSE) terhadap nilai aktual dan prediksi. Berikut merupakan formula RMSE yang digunakan:

<img src="https://media.geeksforgeeks.org/wp-content/uploads/20200622171741/RMSE1.jpg" >

Dengan memplotkan skor RMSE pada saat pelatihan dan validasi di setiap epoch, diperoleh grafik performa model rekomendasi collaborative filtering berbasis deep learning seperti di bawah ini.

<img src="https://user-images.githubusercontent.com/61647791/165275987-5a0bc278-06b6-41eb-af60-6d9304dcb954.png">

RMSE yang diperoleh pada saat pelatihan maupun validasi di epoch terakhir berturut-turut sebesar 0.2700 dan _0.2698. Eror ini tergolong rendah sehingga model dikatakan layak.

## Daftar Pustaka

**Paper:**

Lops, P., de Gemmis, M. and Semeraro, G. (2010), "Content-based Recommender Systems: State of the Art and Trends", Recommender Systems Handbook, pp. 73-105.

**Website:**

Gulati, V. (2022), "Top 10 Countries where Anime is Most Popular and Why!", Epic Dope, available at: https://www.epicdope.com/top-10-countries-where-anime-is-most-popular-and-why/ (accessed 21 April 2022).

https://www.dicoding.com

[Back to Contents](#contents)
