# CardioRisk-AI 🫀

**CardioRisk-AI** is an end-to-end Machine Learning clinical decision-support system designed to predict the likelihood of cardiovascular disease based on standard patient demographic and clinical parameters.

### 🌟 Project Overview

Cardiovascular diseases remain a leading cause of global mortality. Detecting potential heart-related problems early is critical for effective treatment. This project utilizes a **Random Forest Classifier** trained on clinical data to provide healthcare professionals with a data-driven, rapid secondary opinion.

The project features a **Layered Architecture** with a robust data pipeline and is deployed via a highly interactive **Streamlit** user interface.

### 🚀 Features

* **Predictive Modeling:** Achieves high recall to minimize dangerous false negatives in medical diagnosis.
* **Interactive UI:** A clean, professional Streamlit web application for rapid patient data entry.
* **Strict Validation:** Ensures data schema consistency via an enforced business logic layer.
* **Privacy-First:** Processes all clinical data entirely in-memory with zero persistent database storage.

### 🛠️ Technology Stack

* **Frontend:** Streamlit
* **Backend / Logic:** Python (`utils.py`)
* **Machine Learning:** Scikit-Learn (Random Forest)
* **Data Processing:** Pandas, NumPy
* **Model Serialization:** joblib

### ⚙️ How to Run Locally

1. Clone the repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

### ⚠️ Disclaimer

*This system is strictly for educational and decision-support demonstration purposes. It is not clinically validated and should not be used as a replacement for professional medical diagnosis.*