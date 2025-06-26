# STRUKTUR DAFTAR ISI SKRIPSI
# ANALISIS PERBANDINGAN ALGORITMA K-MEANS DAN DBSCAN UNTUK SEGMENTASI PEMAIN MOBILE LEGENDS: BANG BANG DI WILAYAH MEDAN

## BAB II - LANDASAN TEORI

### 2.1 Konsep Dasar Clustering
2.1.1 Definisi dan Tujuan Clustering
2.1.2 Jenis-jenis Algoritma Clustering
2.1.3 Tantangan dalam Data Game Analytics
2.1.4 Aplikasi Clustering dalam Gaming Industry

### 2.2 Algoritma K-Means
2.2.1 Prinsip Kerja dan Formulasi Matematika
2.2.2 Algoritma Lloyd's dan Konvergensi
2.2.3 Penentuan Jumlah Cluster Optimal (Elbow Method, Silhouette Analysis)
2.2.4 Kelebihan dan Keterbatasan K-Means
2.2.5 Aplikasi K-Means dalam Esports Analytics

### 2.3 Algoritma DBSCAN (Density-Based Spatial Clustering)
2.3.1 Prinsip Clustering Berbasis Kepadatan
2.3.2 Parameter Epsilon (ε) dan MinPoints
2.3.3 Konsep Core Points, Border Points, dan Noise Points
2.3.4 Algoritma DBSCAN dan Kompleksitas Komputasi
2.3.5 Keunggulan DBSCAN dalam Deteksi Outlier

### 2.4 Metrik Evaluasi Clustering
2.4.1 Internal Validation Metrics
   - Silhouette Score: S(i) = (b(i) - a(i)) / max(a(i), b(i))
   - Calinski-Harabasz Index: CH = (SSB/SSW) × ((n-k)/(k-1))
   - Davies-Bouldin Index: DB = (1/k) Σ max((σᵢ + σⱼ)/d(cᵢ,cⱼ))
2.4.2 Statistical Significance Testing
   - ANOVA F-Test: F = MSB/MSW
   - Kruskal-Wallis Test: H = (12/N(N+1)) Σ(Rᵢ²/nᵢ) - 3(N+1)
2.4.3 Stability and Robustness Analysis
   - Cross-validation untuk Clustering
   - Feature Importance Analysis: F-ratio = σ²between/σ²within
2.4.4 Comparative Metrics
   - Adjusted Rand Index (ARI)
   - Normalized Mutual Information (NMI)

### 2.5 Mobile Legends: Bang Bang sebagai Domain Penelitian
2.5.1 Karakteristik Game MOBA (Multiplayer Online Battle Arena)
2.5.2 Sistem Role dan Hero Classification
2.5.3 Metrik Performa Pemain (KDA, Win Rate, GPM)
2.5.4 Dinamika Permainan dan Player Behavior

### 2.6 Penelitian Terkait Player Segmentation
2.6.1 Esports Player Typology (Hedlund, 2021)
2.6.2 Game Behavior Data Clustering (Drachen et al., 2014)
2.6.3 MOBA Player Profiling Studies
2.6.4 Gap Analysis dan Posisi Penelitian

## BAB III - ANALISIS DAN PERANCANGAN SISTEM

### 3.1 Kerangka Kerja Penelitian
3.1.1 Metodologi Penelitian Kuantitatif Komparatif
3.1.2 Tahapan Penelitian dan Timeline
3.1.3 Hipotesis Penelitian
   - H1: DBSCAN menghasilkan cluster yang lebih natural dibanding K-Means
   - H2: DBSCAN lebih efektif dalam deteksi outlier pemain
   - H3: Kombinasi metrik evaluasi memberikan assessment yang komprehensif

### 3.2 Arsitektur Sistem Analisis
3.2.1 Use Case Diagram Sistem Clustering
   - Actor: Data Analyst, Researcher
   - Use Cases: Load Data, Preprocess Features, Execute K-Means, Execute DBSCAN, 
     Evaluate Clusters, Compare Algorithms, Generate Reports
3.2.2 System Architecture Overview
3.2.3 Data Flow Diagram Level 0 dan Level 1
3.2.4 Technology Stack dan Framework

### 3.3 Pengumpulan dan Sumber Data
3.3.1 Data Primer: Survey Data Pemain Medan (245 records)
   - Sampling Method dan Justifikasi Ukuran Sample
   - Data Collection Protocol
3.3.2 Data Sekunder: Gaming Analytics (Simulasi 1000 records)
   - Rank Distribution Data
   - Hero Meta Analytics
3.3.3 Data Validation dan Quality Assurance

### 3.4 Perancangan Dataset dan Feature Engineering
3.4.1 Primary Dataset Structure
   - Player Performance Metrics: win_rate, avg_kda, match_frequency
   - Role Classification: role_Fighter, role_Mage, role_Marksman, dll
3.4.2 Feature Selection dan Justifikasi
3.4.3 Data Integration Strategy (Primary + Secondary)
3.4.4 Handling Missing Values dan Outliers

### 3.5 Preprocessing dan Normalisasi Data
3.5.1 Data Cleaning Protocol
3.5.2 Feature Scaling dengan MinMaxScaler
3.5.3 Dimensionality Analysis
3.5.4 Data Splitting untuk Validation

### 3.6 Implementasi Algoritma Clustering
3.6.1 K-Means Implementation
   - Parameter Tuning (n_clusters: 2-10)
   - Initialization Methods (k-means++, random)
   - Convergence Criteria
3.6.2 DBSCAN Implementation
   - Epsilon Parameter Optimization
   - MinPoints Parameter Selection
   - Distance Metrics Selection
3.6.3 Algorithm Optimization dan Performance Tuning

### 3.7 Framework Evaluasi Komprehensif
3.7.1 Internal Validation Metrics Implementation
3.7.2 Statistical Testing Framework
3.7.3 Cross-validation Strategy
3.7.4 Comparative Analysis Protocol

## BAB IV - IMPLEMENTASI DAN PENGUJIAN

### 4.1 Lingkungan Pengembangan dan Tools
4.1.1 Development Environment Setup
4.1.2 Library dan Dependencies (scikit-learn, pandas, numpy)
4.1.3 Hardware Specifications dan Performance Benchmarks

### 4.2 Eksplorasi Data Awal (Exploratory Data Analysis)
4.2.1 Descriptive Statistics Pemain Mobile Legends Medan
4.2.2 Distribusi Features dan Correlation Analysis
4.2.3 Role Distribution dan Player Characteristics
4.2.4 Data Visualization dengan PCA dan t-SNE

### 4.3 Parameter Optimization dan Tuning
4.3.1 K-Means Cluster Number Optimization
   - Elbow Method Implementation
   - Silhouette Analysis untuk Optimal K
   - Gap Statistic Method
4.3.2 DBSCAN Parameter Tuning
   - Epsilon Selection dengan k-distance graph
   - MinPoints Sensitivity Analysis
   - Parameter Grid Search

### 4.4 Hasil Clustering dan Analisis
4.4.1 K-Means Clustering Results
   - Cluster Centers dan Interpretasi
   - Cluster Size Distribution
   - Player Profiles per Cluster
4.4.2 DBSCAN Clustering Results
   - Natural Cluster Formation
   - Noise Points Analysis (Outlier Detection)
   - Cluster Density Analysis
4.4.3 Visual Comparison dengan Scatter Plots

### 4.5 Evaluasi Komprehensif dengan 8 Metrik Utama
4.5.1 Internal Validation Results
   - Silhouette Score Analysis: K-Means vs DBSCAN
   - Calinski-Harabasz Index Comparison
   - Davies-Bouldin Index Assessment
4.5.2 Statistical Significance Testing
   - ANOVA F-Test Results per Feature
   - Kruskal-Wallis Test untuk Non-parametric Analysis
4.5.3 Advanced Metrics Analysis
   - Inertia dan Cluster Compactness
   - Inter-cluster Distance Analysis
   - Feature Importance Ranking

### 4.6 Stability dan Robustness Analysis
4.6.1 Cross-validation Results
4.6.2 Algorithm Stability Assessment
4.6.3 Parameter Sensitivity Analysis
4.6.4 Performance Scalability Testing

### 4.7 Comparative Analysis K-Means vs DBSCAN
4.7.1 Quantitative Comparison Results
4.7.2 Qualitative Assessment
4.7.3 Use Case Recommendations
4.7.4 Performance Trade-offs Analysis

### 4.8 Player Segmentation Results
4.8.1 Mobile Legends Player Archetypes Identification
4.8.2 Role-based Clustering Insights
4.8.3 Outlier Players Analysis (DBSCAN Advantage)
4.8.4 Practical Applications untuk Game Analytics

### 4.9 Validation dan Verification
4.9.1 Result Validation dengan Domain Experts
4.9.2 Statistical Significance Confirmation
4.9.3 Reproducibility Testing
4.9.4 Limitations dan Threats to Validity

## BAB V - KESIMPULAN DAN SARAN

### 5.1 Kesimpulan

**5.1.1 Jawaban Rumusan Masalah Utama**
Berdasarkan analisis komprehensif terhadap 245 pemain Mobile Legends di wilayah Medan menggunakan 8 metrik evaluasi clustering, penelitian ini berhasil menjawab permasalahan komparasi algoritma K-Means dan DBSCAN untuk segmentasi pemain game MOBA:

**5.1.2 Keunggulan Relatif Algoritma**
- K-Means: Unggul dalam konsistensi hasil, computational efficiency, dan interpretability cluster centers
- DBSCAN: Superior dalam natural cluster formation, outlier detection, dan handling non-spherical clusters

**5.1.3 Kontribusi Penelitian**
- Pertama kali membandingkan K-Means vs DBSCAN pada data pemain Mobile Legends Indonesia
- Implementasi framework evaluasi 8 metrik statistik untuk clustering gaming data
- Identifikasi player archetypes berbasis data objektif performa gameplay

**5.1.4 Validitas Hipotesis Penelitian**
- H1: Terkonfirmasi - DBSCAN menghasilkan cluster natural dengan Silhouette Score lebih tinggi
- H2: Terkonfirmasi - DBSCAN mendeteksi 5 outlier players yang tidak teridentifikasi K-Means
- H3: Terkonfirmasi - Kombinasi 8 metrik memberikan assessment komprehensif dan robust

### 5.2 Implikasi Praktis
5.2.1 Rekomendasi untuk Game Analytics Industry
5.2.2 Aplikasi dalam Player Retention Strategies
5.2.3 Framework untuk Developer Mobile Legends

### 5.3 Keterbatasan Penelitian
5.3.1 Scope Geografis (Medan only)
5.3.2 Sample Size Considerations
5.3.3 Temporal Limitations

### 5.4 Saran Penelitian Mendatang
5.4.1 Ekspansi ke Dataset Multi-regional
5.4.2 Implementasi Deep Learning Clustering
5.4.3 Real-time Player Segmentation System
5.4.4 Integration dengan Recommendation Systems

---

## HIGHLIGHTS STRUKTUR INI:

**1. Comprehensive Coverage:** Mencakup semua 8 perhitungan statistik yang diimplementasikan
**2. Academic Rigor:** Mengacu pada 10 penelitian relevan dengan gap analysis yang jelas
**3. Use Case Diagram:** Ditempatkan di Bab 3.2.1 sesuai saran dosen
**4. Clear Problem Resolution:** Kesimpulan langsung menjawab rumusan masalah utama
**5. Research Validity:** Framework hipotesis testing yang kuat
**6. Practical Applications:** Menghubungkan hasil akademis dengan implementasi industri