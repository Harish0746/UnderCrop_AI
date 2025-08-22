import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#Dataset 
file = pd.read_csv("crops_dataset.csv")

#top 5 most frequent crops
top_crops = file["Crop"].value_counts().nlargest(5).index
file = file[file["Crop"].isin(top_crops)]

#Feature Engineering
file["ProfitPotential"] = file["AvgYield"] * file["AvgMandiPrice"]
file["SmartScore"] = file["MarketScore"] * file["ResilienceScore"]

# Normalizing numeric columns
numeric_cols = ["ProfitPotential", "SmartScore"]
scaler = MinMaxScaler(feature_range=(0,100))
file[numeric_cols] = scaler.fit_transform(file[numeric_cols])

#target crop encoding
L_crop = LabelEncoder()
file["Croplabel"] = L_crop.fit_transform(file["Crop"])

# One-hot encode categorical features
file = pd.get_dummies(file, columns=["SoilType","State","IrrigationNeeded"])

# Features & target (modify)
X = file[numeric_cols + [col for col in file.columns if col.startswith(("SoilType_", "State_", "IrrigationNeeded_"))]]
y = file["Croplabel"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# RandomForest model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

print("===  UnderCrop AI: Crop Recommendation ===")

# User Input 
soil = input("Enter Soil Type (Loamy/Clay/Sandy/Alluvial/Red/Black): ").strip()
state = input("Enter State (Tamil Nadu/Maharashtra/Punjab): ").strip()
irrigation = input("Enter Irrigation Needed (High/Medium/Low): ").strip()

# ====== Prepare Sample Row ======
sample_data = {}

# Numeric features: use average values from training set
for col in numeric_cols:
    sample_data[col] = file[col].mean()  # simple default for hackathon

# One-hot encoding: mappin user input to correct columns(?)
for col in X.columns:
    if col.startswith("SoilType_"):
        sample_data[col] = 1 if col.split("_")[1] == soil else 0
    elif col.startswith("State_"):
        sample_data[col] = 1 if col.split("_")[1] == state else 0
    elif col.startswith("IrrigationNeeded_"):
        sample_data[col] = 1 if col.split("_")[1] == irrigation else 0

# Fill missing columns with 0
sample_df = pd.DataFrame(columns=X.columns)
for col in X.columns:
    sample_df.at[0, col] = sample_data.get(col, 0)

# prediction
probs = model.predict_proba(sample_df)[0]
top5_index = probs.argsort()[-5:][::-1]
top5_crops = L_crop.inverse_transform(top5_index)
top5_scores = probs[top5_index]*100  # converting to 0-100 for demo

# output
print("\n Recommended Crops:")
for crop, score in zip(top5_crops, top5_scores):
    print(f"- {crop} (Suitability Score: {score:.1f}%)")
# ... (your existing code)
 
