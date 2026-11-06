# 🧲 Machine Learning Predictive Framework for Localized Oral Cancer Treatment
> **An Interdisciplinary Data Engineering Pipeline Digitising Preclinical Magnetic Hyperthermia Therapy Data**

## 📄 Core Research Reference
This computational pipeline is built directly upon the experimental metrics and parameter bounds established in my peer-reviewed published research:
* **Paper Title:** *Preclinical validation of magnetic hyperthermia therapy for locally advanced oral cancer treatment*

---

## 🎯 Project Objective
The goal of this project is to translate preclinical localized thermal ablation data into a dynamic, predictive computational framework. By converting visual experimental graphs from the publication's MTT viability assay (Fig. 4a) into a digitized data matrix, this system trains an ensemble **Random Forest Classifier** to predict the therapeutic success of localized cell suppression based on two primary variables: **Incubation Time** and **Nanoparticle Concentration**.

By mapping these biological relationships into an object-oriented Python architecture, this project serves as a proof-of-concept for simulating experimental therapeutic outcomes computationally before initiating physical, manual laboratory trials.

---

## 🛠️ System Architecture & Code Breakdown
The project is built using professional, industry-standard **Object-Oriented Programming (OOP)** patterns, cleanly splitting data handling from algorithmic calculation:

### 1. Data Matrix Layer (`data/hyperthermia_dataset.csv`)
* **Features ($X$):** Contains digitized values mapping `incubation_time_hr` (24, 48, 72 hours) and `concentration_mg_ml` (ranging from 0 to 15 mg/mL).
* **Target ($y$):** Evaluates `optical_density` (OD) metrics representing cell viability. The pipeline applies a binary classification rule where an Optical Density drop below 0.08 is mapped as a `1` (Successful therapeutic cell suppression/death), and anything above is mapped as a `0` (Control baseline/insufficient suppression).

### 2. Algorithmic Machine Learning Layer (`src/model_engine.py`)
* **Data Partitioning:** Automatically segments the 27 digitized experimental samples into an **80% training set** (to teach the algorithm patterns) and a **20% testing set** (to evaluate validation accuracy on unseen data). It utilizes *stratification* to ensure binary class proportions remain perfectly balanced across both splits.
* **The Ensemble Model:** Deploys a **Random Forest Classifier** composed of 100 decision trees. This model was chosen because biological and nanoparticle dose-response datasets exhibit highly complex, non-linear relationships. Random Forests protect the compact dataset from "overfitting" (memorizing noise or outliers in small sample sizes).

### 3. Orchestration Control (`main.py`)
* Acts as the centralized system switchboard that coordinates the automated ingestion of the CSV file, triggers binary classification boundaries, drops raw leakage metrics, runs the model evaluation loops, and exports diagnostics.

---

## 📊 Evaluation Outputs & Strategic Insights
When executed, the system outputs detailed predictive diagnostic reports (Precision, Recall, and F1-Scores) and automatically exports a localized visualization chart titled **`feature_importance.png`**.

This feature importance map ranks exactly which laboratory control variable had the highest mathematical influence on driving successful therapeutic outcomes. This bridges physical bench sciences with computational data structures, verifying whether the machine learning engine's algorithmic splits align with observed biological cancer cell suppression mechanisms.

