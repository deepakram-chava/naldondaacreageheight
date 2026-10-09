# DATASET SOURCE & PROVENANCE SPECIFICATION
## 10-Class Paddy Disease Classification Benchmark

### 1. Dataset Identification & Provenance
- **Dataset Name**: Paddy Disease Classification (10-Class Benchmark Dataset)
- **Original Research Source**: *Paddy Doctor: A Visual Image Dataset for Automated Paddy Disease Classification and Benchmarking*
- **Original Authors & Institutional Affiliation**:
  - Petchiammal A., Manonmaniam Sundaranar University, Tirunelveli, Tamil Nadu, India
  - Briskline Kiruba S., Manonmaniam Sundaranar University
  - Murugan D., Manonmaniam Sundaranar University
  - Pandarasamy Arjunan, Robert Bosch Center for Cyber-Physical Systems, Indian Institute of Science (IISc), Bangalore, India
- **Original Source Publication**: IEEE Dataport / Kaggle Competition (*Paddy Disease Classification*)
  - Official Kaggle: https://www.kaggle.com/c/paddy-disease-classification
  - IEEE Dataport DOI: [10.21227/hz4v-af08](https://dx.doi.org/10.21227/hz4v-af08)
- **Verified Download Source**: Hugging Face Datasets (`techharry/paddy_disease_classification`)
  - URL: https://huggingface.co/datasets/techharry/paddy_disease_classification
- **License**: CC BY 4.0 (Creative Commons Attribution 4.0 International)
- **Download Date**: 2026-10-05

### 2. Scholarly Citation
```bibtex
@article{petchiammal2022paddy,
  title={Paddy Doctor: A Visual Image Dataset for Automated Paddy Disease Classification and Benchmarking},
  author={Petchiammal, A and Briskline Kiruba, S and Murugan, D and Arjunan, Pandarasamy},
  journal={IEEE Dataport},
  year={2022},
  doi={10.21227/hz4v-af08}
}
```

### 3. Class Taxonomy (Strictly 10 Classes)
This dataset contains exactly 10 distinct classes of paddy leaf health and disease states:

| Class ID | Class Name | Description / Pathogen |
| :---: | :--- | :--- |
| 0 | `bacterial_leaf_blight` | *Xanthomonas oryzae pv. oryzae* (bacterial wilt/blight lesions) |
| 1 | `bacterial_leaf_streak` | *Xanthomonas oryzae pv. oryzicola* (interveinal translucent lesions) |
| 2 | `bacterial_panicle_blight` | *Burkholderia glumae* (panicle discoloration and rotting) |
| 3 | `blast` | *Magnaporthe oryzae* (spindle-shaped necrotic lesions) |
| 4 | `brown_spot` | *Bipolaris oryzae* (oval brown spots with gray center) |
| 5 | `dead_heart` | Stem borer insect damage (*Scirpophaga incertulas*) |
| 6 | `downy_mildew` | *Sclerophthora macrospora* (chlorotic yellow-green streaking) |
| 7 | `hispa` | Rice hispa beetle damage (*Dicladispa armigera*) |
| 8 | `normal` | Healthy paddy leaf (disease-free control) |
| 9 | `tungro` | Rice tungro spherical & bacilliform virus |

*Note: No classes have been merged, renamed, omitted, or syntheticized. Larger multi-task or expanded 12/20-class variants were strictly excluded per specification.*

### 4. Storage & Processing Protocol
- **Raw Archive Location**: `data/raw/paddy_disease/`
  - Unmodified binary Parquet files: `train-00000-of-00002.parquet` and `train-00001-of-00002.parquet`.
  - Raw images remain completely untouched as immutable ground truth.
- **Processed Benchmark Directory**: `dataset/`
  - Organized strictly into the 10 class subdirectories.
  - Image files saved in lossless RGB JPEG format.

### 5. Integrity & Preprocessing Record
- **Missing Classes**: 0 (all 10 classes present).
- **Corrupted Images**: 0 (all 10,407 image streams validated via PIL image header decoding).
- **Duplicate Images**: 72 exact MD5 checksum groups identified (representing 74 redundant images out of 10,407).
- **Image Extensions**: 100% `.jpg` (10,407 files).
- **Color Format**: 100% 3-channel RGB.
- **Image Dimensions**: Standard 480x640 (10,403 images) and 640x480 (4 images).
- **Modifications**: Zero synthetic data, zero interpolated files.
- **Separation from Field Validation**:
  - In-situ Nalgonda NRSC/ISRO field survey photos and records (`survey_0.csv`) are strictly quarantined from this training benchmark dataset.

