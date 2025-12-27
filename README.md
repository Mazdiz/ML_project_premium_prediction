# Health Insurance Premium Prediction 🏥

[Live Demo](https://ml-regression-project-premium-prediction.streamlit.app)


## 📌 Overview
This project predicts health insurance premiums using machine learning. It fulfills the requirements for a modular Python application, demonstrating a full data science workflow: data cleaning, feature engineering, regression modeling (XGBoost), and deployment via Streamlit.


## 🚀 Key Features
Modular Architecture: Organized into a Main Module (main.py) for the UI and a Logic Module (prediction_helper.py) for processing.

Data Handling: Uses Dictionaries and Lists to manage feature inputs and categorical encoding mapping.

Numeric Processing: Implements robust type conversion (e.g., float, int) to handle user inputs from the Streamlit interface.

Deployment: Interactive Streamlit app for real-time user input and instant premium prediction.


## 💻 Tech Stack
Language: Python 3.13

Environment: Miniconda (/opt/miniconda3/bin/python)

Libraries: Pandas, NumPy, Scikit-learn, XGBoost, Streamlit

Tools: Git, GitHub, Jupyter Notebooks


## 🛠️ Installation & Usage
1. Clone the repo

Bash
git clone https://github.com/Mazdiz/insurance-premium
cd insurance-premium

2. Install dependencies

Note: Use the specific path to your Miniconda environment to ensure the correct versions are installed.

Bash
/opt/miniconda3/bin/python -m pip install -r requirements.txt

3. Run the app

Bash
/opt/miniconda3/bin/python -m streamlit run main.py


## 📂 Project Structure
main.py: The entry point of the application; handles the Streamlit frontend.

prediction_helper.py: The backend logic module that loads the model and processes data.

Artifacts/: Contains the pre-trained XGBoost model and scaling objects.

Insurance_Analysis.ipynb: The complete notebook containing EDA and model training logic.


## 📊 Insights
Feature Importance: Age, BMI, and smoking status are the strongest drivers of premium costs.

## Optimization: The XGBoost model was patched for compatibility with Python 3.13 to ensure stable runtime performance.


## 🤝 Contact
•	LinkedIn: [Prudence Mpieri](https://www.linkedin.com/in/prudencempieri)
•	Github: [Mazdiz](https://github.com/Mazdiz)
