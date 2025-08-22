# 🌱 UnderCrop AI – Crop Recommendation Engine  

UnderCrop AI is a lightweight machine learning–powered system that helps farmers and agri-entrepreneurs identify the **most suitable crops** based on soil type, state, irrigation needs, and profitability.  

This hackathon MVP uses a **RandomForestClassifier** trained on a curated crop dataset with engineered features like profit potential and resilience score.  

---

## ⚡ Features
- 📊 Uses real **agri dataset** (`crops_dataset.csv`)  
- 🌾 Focuses on **top 5 frequent crops** for efficient demo performance  
- 💰 Computes **Profit Potential** (Yield × Mandi Price)  
- 🧮 Introduces **Smart Score** (Market × Resilience)  
- ⚖️ Normalizes scores to a 0–100 scale  
- 🤖 Trains a **RandomForest Classifier** with categorical one-hot encoding  
- 🎯 Outputs **Top 5 Recommended Crops** with suitability scores  

---

## 📂 Project Structure

<pre>
UnderCrop-AI/
│── crops_dataset.csv # Sample synthetic dataset
│── undercrop_rf.py # Main RandomForest crop recommender
│── README.md # Documentation
</pre>

## 🛠 Installation & running
1. Clone this repo:
   ```bash
   git clone https://github.com/harish0746/UnderCrop-AI.git
   cd UnderCrop-AI
   ```
2. Create a virtual environment & install dependencies:
   ```bash
    python -m venv venv
    source venv/bin/activate   # Mac/Linux
    .\venv\Scripts\activate    # Windows

    pip install pandas scikit-learn
   ```
3. run the recommender program 
   ```bash
   python model.py
   ```
4. Provide with necessary inputs and witness the recommendation system working.