# Repo for UCon-SFDA

This will be the official git repo for our paper "Revisiting Source-Free Domain Adaptation: a New Perspective via Uncertainty Control".

---
## Get Started

### 1. Data Preperation

---
### 2. Model Preperation

Models:
- shot: office-home partial set; visda2017; visdaRust
- nrc: office-home
- ucon_sfda: office31, domainnet126

- generalDA: ucon_sfda (office31, domainnet126) + nrc (office-home) + shot's visda2017 
- specialCase: office-home partial set and visdaRust
---
### 3. Target Adaptation


#### 3.1. UCon_SFDA - basic methodology

#### 3.2. autoUCon_SFDA - auto-hyperparam selection version

- cal version
- stat version

---

## 注意事项

1. 最后数据文件的名字要和common/vision/datasets对齐