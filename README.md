# 🌍 Lake Monitoring System
Long-term lake monitoring is an important hydrology and climate research area
to understand lake surface area dynamics in response to global environmental
change. The study implements an automated analysis framework to monitor 100
lakes within a user-specified country over the years 2000-2026 with the systematic
integration and application of remote sensing, deep learning, statistical trend
and explainable AI techniques. The proposal enables a consistent longterm
dataset acquisition using the central cloud-based Google Earth Engine
platform to support the extraction of global lakes and climate metadata, multisource
satellite and Global Surface Water (GSW) mask imagery. A weighted
soft-voting ensemble of three modern deep learning architectures including
UNet++, SegFormer and DeepLab facilitates a robust multi-model lake-water
segmentation, offering performance improvement over single-model baselines.
Threshold tuning and ensemble-weight calibration process aimed at best F1-score
ensures optimal segmentation configuration. The ensemble prediction results
drive the preparation of lake surface area records for subsequent lake area-climate
analysis. The linear regression identifies statistical trend of long-term variations
in lake surface area and climate drivers with associated trend severity. Gradient
Boosting Regression based Permutation Importance and SHAP analysis provides
interpretable insights of how environmental factors explain variations in lake area
dynamics.

The Canada lakes analysis results identify both lake area expansion
and reduction over the study period, with more frequent increasing area trends.
Temperature and precipitation act as the most influencing climate drivers to
observed lake area dynamics, with temperature at positive and precipitation at
negative association.

---

## 📂 Repository Structure

### Scripts
```
msc_2026_lakeobservation/
├──Lake_Dataset/                   
│ ├── extract_lake_dimension.py
│ ├── download_lake_dataset.py
│ ├── download_GSW_mask.py
│ └── extract_GSW_mask.py
├──Lake_Segmentation/
│ ├── split_dataset.py 
│ ├── train_unet++.py
│ ├── train_segformer.py
│ ├── train_deeplab.py
│ ├── evaluate_models.py
│ └── plot_segmentation.py
├──Lake_Area_Climate_Relationship
│ ├── extract_lake_area.py 
│ ├── extract_climate.py
│ ├── evaluate_lake_trend.py
│ ├── plot_trend.py
│ └── analyse_climate_area.py
```
### Runtime Data
``` 
msc_2026_lakeobservation/
├──data_images/                   
├──mask_images/
├──download GSW data/                   
├──model/
│ ├── unet++_model_final.keras
│ ├── segformer_model_final.keras
│ └── deeplab_model_final.keras
├──predictions/                   
├──mask_images/
├──trend/   
│ ├── lake_area.py
│ ├── lake_climate.py
│ ├── lake_area_trend.py
│ └── lake_trend.py                
├──plots/
│ ├── Fig_lake_segmentation.png
│ ├── Fig_lake_trend.png
│ ├── Fig_PI_Test.png
│ ├── Fig_PI_Category.png
│ ├── Fig_SHAP_Test.png
│ └── Fig_SHAP_Category.py

```

## Google Colab Environment

---
```bash
!git clone https://oauth2:$GIT_TOKEN@scc-source.lancs.ac.uk/ladp/msc_2026_lakeobservation.git
```

### ⚙️ Installation
```bash
!pip install earthengine-api geemap -q

!pip install geedim

!pip install rasterio matplotlib tqdm

!pip install -U geemap

!pip install keras-unet-collection

!pip install keras_cv

!pip install tensorflow opencv-python
```

### Stage 1: Lake Datatset Preparation
**Lake Metadata Extraction**
```bash
%run Lake_Dataset/extract_lake_dimension.py
```

**Multi-source Satellite Lake Imagery Extraction**
```bash
%run -i Lake_Dataset/download_lake_dataset.py
```

**Global Surface Water (GSW) Mask Imagery Extraction**
```bash
%run -i Lake_Dataset/download_GSW_dataset.py
%run -i Lake Dataset/extract_GSW_mask.py
```

### Stage 2: Lake Segmentation
**Dataset Train-Validation-Test Split**
```bash
%run -i Lake_Segmentation/split_dataset.py
```

**Deep Learning Models Training**
```bash
%run -i Lake_Segmentation/train_unet++.py
%run -i Lake_Segmentation/train_segformer.py
%run -i Lake_Segmentation/train_deeplab.py
```

**Model Evaluation and Ensemble**
```bash
%run -i Lake_Segmentation/evaluate_models.py
```

**Segmentation Plot**
```bash
%run -i Lake_Segmentation/plot_segmentation.py
```

### Stage 3: Lake Area-Climate Relationship
**Lake Area Estimation**
```bash
%run -i Lake_Area_Climate_Analysis/extract_lake_area.py
```

**Climate Variables Extraction**
```bash
%run -i Lake_Area_Climate_Analysis/extract_climate.py
```

**Lake Area-Climate Trend Evaluation**
```bash
%run -i Lake_Area_Climate_Analysis/evaluate_lake_trend.py
%run -i Lake_Area_Climate_Analysis/plot_trend.py
```

**Lake Area-Climate Relationship Analysis**
```bash
%run -i Lake_Area_Climate_Analysis/analyse_climate_area.py
```