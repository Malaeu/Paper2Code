# Medical Dataset for Gender-Specific Analysis

## Overview
This dataset contains medical records from a longitudinal study of patients with different gender, demographic information, and various clinical measurements. The dataset is designed to support the development of gender-specific predictive models for liver disease progression.

## Dataset Structure

### Demographics
- `gender`: Gender of the patient (male/female)
- `age`: Age in years
- `bmi`: Body mass index
- `smoking_status`: Whether the patient is a current smoker (yes/no)

### Clinical Variables
- `diabetes`: Whether the patient has diabetes (yes/no)
- `hypertension`: Whether the patient has hypertension (yes/no)
- `systolic_bp`: Systolic blood pressure in mmHg
- `heart_rate`: Heart rate in bpm

### Laboratory Values
- `blood_marker_a`: A liver-specific blood marker (continuous value)
- `blood_marker_b`: Another liver-specific blood marker (continuous value)
- `ast`: Aspartate aminotransferase level
- `alt`: Alanine aminotransferase level
- `albumin`: Albumin level in g/dL
- `platelet`: Platelet count
- `inr`: International normalized ratio

### Outcome Variables
- `event`: Whether the patient developed liver failure (0=no, 1=yes)
- `time_to_event`: Time in months until liver failure occurred or until last follow-up
- `followup_time`: Total follow-up time in months

## Cohort Information
- `dataset_a`: Patients from research center A (0=no, 1=yes)
- `dataset_b`: Patients from research center B (0=no, 1=yes)

## Statistics
- Total number of patients: 3,500
- Female patients: 1,820 (52%)
- Male patients: 1,680 (48%)
- Median follow-up time: 48 months
- Total events: 560 (16%)

## Special Considerations
1. The dataset includes patients with at least 12 months of follow-up
2. Early events (within first 12 months) have been censored
3. Missing data has been imputed using multiple imputation
4. The dataset contains some patients with pre-existing conditions that may influence outcomes

## Research Question
The primary research question is whether gender-specific predictive models for liver disease progression demonstrate better performance than non-stratified models, and which variables contribute most significantly to risk prediction in each gender.