# AI-Based Career Recommendation System

# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.metrics import accuracy_score
import streamlit as st

# Step 2: Simulated dataset
# You can replace this with real data or expand this dataset
sample_data = [
    {"skills": ["Python", "ML", "Statistics"], "gpa": 8.5, "domain": "Data Science", "career": "Data Scientist"},
    {"skills": ["Java", "DSA", "OOP"], "gpa": 7.5, "domain": "Software", "career": "Software Developer"},
    {"skills": ["SQL", "Excel", "Business"], "gpa": 7.0, "domain": "Business", "career": "Business Analyst"},
    {"skills": ["Deep Learning", "Python", "AI"], "gpa": 8.8, "domain": "AI", "career": "AI Engineer"},
    {"skills": ["C++", "OOP", "DSA"], "gpa": 6.5, "domain": "Software", "career": "Software Developer"},
    {"skills": ["Python", "Data Analysis", "SQL"], "gpa": 7.8, "domain": "Data", "career": "Data Analyst"},
    {"skills": ["Excel", "Power BI", "Presentation"], "gpa": 7.9, "domain": "Business", "career": "Business Analyst"}
]

df = pd.DataFrame(sample_data)

# Step 3: Preprocessing
mlb = MultiLabelBinarizer()
skill_matrix = mlb.fit_transform(df['skills'])
skill_df = pd.DataFrame(skill_matrix, columns=mlb.classes_)
df = pd.concat([df, skill_df], axis=1)
df.drop(columns=['skills'], inplace=True)

domain_encoded = pd.get_dummies(df['domain'])
df = pd.concat([df, domain_encoded], axis=1)
df.drop(columns=['domain'], inplace=True)

X = df.drop(columns=['career'])
y = df['career']

scaler = StandardScaler()
X[['gpa']] = scaler.fit_transform(X[['gpa']])

# Step 4: Model Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)
pred = model.predict(X_test)

# Step 5: Evaluation
accuracy = accuracy_score(y_test, pred)

# Step 6: Streamlit App
st.title("🎓 AI-Based Career Recommendation System")
st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

# Skill input
user_skills = st.multiselect("Select your skills:", options=mlb.classes_)
user_gpa = st.slider("Enter your GPA:", 0.0, 10.0, 7.5)
user_domain = st.selectbox("Preferred Domain:", options=domain_encoded.columns)

if st.button("Predict Career"):
    # Process input
    input_data = {skill: 1 if skill in user_skills else 0 for skill in mlb.classes_}
    for dom in domain_encoded.columns:
        input_data[dom] = 1 if dom == user_domain else 0
    input_data['gpa'] = user_gpa

    input_df = pd.DataFrame([input_data])
    input_df[['gpa']] = scaler.transform(input_df[['gpa']])

    prediction = model.predict(input_df)[0]
    st.success(f"✅ Recommended Career Path: **{prediction}**")
