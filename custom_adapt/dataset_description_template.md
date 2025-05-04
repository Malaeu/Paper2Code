# Dataset Description

## Dataset Overview

**Name**: [Dataset name]
**Version**: [v1.0 — YYYY‑MM‑DD]
**Purpose**: [Brief description of the dataset purpose and what it represents]
**Source / Provenance**: [Data origin — e.g., clinical trial, open repository, IoT sensors]
**Ethical Approval & Consent**: [IRB # / license / GDPR compliance / consent status]
**Size**: [Number of samples / observations and variables]
**Domain**: [Research field or domain]
**Timeframe**: [When the data were collected]

## Data Structure

**Format**: [CSV, Parquet, SQL db, etc.]
**Dimensions**: [Rows × Columns]
**ID Variable**: [Primary key / unique identifier]
**Data Organization**: [One row per patient, nested JSON, longitudinal, etc.]
**Versioning & Updates**: [How versions are tagged, changelog location]

## Variable Descriptions

### Key Variables Table

| Variable Name  | Type        | Description                  | Units / Scale | Allowed Values                  | Missing‑Value Code | Role           | Scientific Equivalents |
| -------------- | ----------- | ---------------------------- | ------------- | ------------------------------- | ------------------ | -------------- | ---------------------- |
| age            | numeric     | Patient age at diagnosis     | years         | 18–90                           | −999               | predictor      | age, patient age       |
| gender         | categorical | Biological sex               | –             | male, female, other             | unknown            | stratification | sex                    |
| treatment      | categorical | Treatment protocol           | –             | standard, experimental, placebo | NULL               | independent    | intervention           |
| survival\_time | numeric     | Time until event / censoring | months        | 0–120                           | N/A                | outcome        | follow‑up time         |
| event          | binary      | Event occurred?              | –             | 0 = no, 1 = yes                 | −1                 | outcome        | death, recurrence      |

### Detailed Variable Information

#### [Variable Name]

* **Full Name**:
* **Description**:
* **Collection Method**:
* **Data Type**:
* **Units**:
* **Range / Levels**:
* **Coding Scheme**:
* **Missing Values**:
* **Transformations / Standardization**:
* **Role in Analysis**:
* **Common Scientific Terms**:
* **Notes**:

*(Repeat for each variable)*

## Variable Relationships

**Primary Outcome(s)**: [...]
**Primary Predictors**: [...]
**Stratification Variables**: [...]
**Confounders / Covariates**: [...]
**Hierarchical Structure**: [e.g., patients → visits → lab tests]

**Key Relationships & Visual Checks**:

* [Variable X] strongly predicts [Variable Y].
* [Variables A, B, C] are highly correlated.
* [Variable Z] mediates the relationship between [X] and [Y].

> *Recommendation*: provide correlation heatmaps or network diagrams to validate these points.

## Data Quality Information

* **Missing Data**: [% missing overall; monotone / non‑monotone patterns]
* **Outliers**: [Detection method & handling]
* **Data Cleaning Performed**: [Steps already applied]
* **Known Biases / Limitations**: [Sampling bias, measurement drift, etc.]

## Measurement Units and Scales

* **Standardization / Normalization Applied**: [z‑score, min‑max, etc.; potential effects on interpretation]
* **Special Units**: [Domain‑specific units]
* **Scale Transformations**: [Log, Box‑Cox, etc.]

### Categorical Variable Levels

* **[Categorical Variable]**: level 1 | level 2 | level 3 ...

## Temporal Aspects

* **Time Variables**: [...]
* **Time Units**: [...]
* **Follow‑up Duration**: [...]
* **Censoring Strategy**: [...]

## Special Considerations

* **Class Imbalance**: [Class proportions; mitigation plans — SMOTE, weighting, focal loss]
* **Handling Guidelines**: [e.g., winsorize extreme lab values]
* **Domain Knowledge Notes**: [Key domain rules, clinical cut‑offs]
* **Privacy & Anonymization**: [Hashing, de‑identification, HIPAA safe‑harbor]
* **Ethical Considerations**: [Fairness, potential misuse, data sharing restrictions]

## Variable Mapping Guide (Paper2Code)

| Scientific Concept | Dataset Variable | Notes                                     |
| ------------------ | ---------------- | ----------------------------------------- |
| survival           | survival\_time   | Time from diagnosis to event or censoring |
| mortality          | event            | 1 = death occurred                        |
| sex                | gender           | Biological sex of patient                 |
| risk score         | risk\_index      | Composite score                           |
| age                | age              | Age at baseline                           |
| treatment group    | treatment        | Assigned therapy arm                      |

### Primary Stratification Mapping

* **Original Stratification**: [e.g., race]
* **Dataset Equivalent**: [e.g., ethnicity]

### Treatment / Intervention Mapping

* **Original Treatments**: [Drug A, Drug B, placebo]
* **Your Dataset Treatments**: [Standard, Experimental, Placebo]

---

*This template aligns with FAIR principles (Findable, Accessible, Interoperable, Reusable) and supports reproducible research by emphasizing provenance, ethical compliance, and clear variable mapping.*