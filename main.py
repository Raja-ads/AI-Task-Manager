import streamlit as st
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

# Setup
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Load Data

def load_data(path):
    try:
        data = pd.read_csv(path)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

task_data_path = '/Users/user/Desktop/Ai-Task-Manager/taskdata.csv'
user_profile_path = '/Users/user/Desktop/Ai-Task-Manager/user_profiles.csv'

df = load_data(task_data_path)
user_profiles = load_data(user_profile_path)

if df.empty or user_profiles.empty:
    st.stop()

#Preprocess Text
def preprocess_text(text):
    text = str(text).lower()
    text = ''.join(ch for ch in text if ch not in string.punctuation)
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

df['clean_text'] = df['description'].apply(preprocess_text)

# Feature Extraction and Models

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['clean_text'])

task_model = MultinomialNB()
task_model.fit(X, df['type'])

priority_model = RandomForestClassifier(n_estimators=100, random_state=42)
priority_model.fit(X, df['priority'])

# Streamlit Setup
st.set_page_config(page_title="AI Task Manager with Profiles", layout="centered")
st.title("📋 AI-Powered Task Manager with User Profiles")

# Initialize Session State
if 'history' not in st.session_state:
    st.session_state.history = []

# user_load 
if 'user_load' not in st.session_state:
    st.session_state.user_load = dict(zip(user_profiles['user_id'], user_profiles['current_load']))

# Assign Task Function
def assign_task(task_type):
    
    eligible = user_profiles[
        user_profiles['skills'].str.contains(task_type, case=False, na=False)
    ]

    if eligible.empty:
        return "No eligible users found"

    
    eligible_loads = {user: st.session_state.user_load.get(user, 0) for user in eligible['user_id']}

    # Assign to user with least load
    return min(eligible_loads, key=eligible_loads.get)

# Task Input Form
with st.form("task_form"):
    task_input = st.text_area("📝 Enter Task Description", height=100)
    submitted = st.form_submit_button("Assign Task")

if submitted and task_input:
    clean_input = preprocess_text(task_input)
    input_vector = tfidf.transform([clean_input])

    predicted_type = task_model.predict(input_vector)[0]
    predicted_priority = priority_model.predict(input_vector)[0]
    assigned_user = assign_task(predicted_type)

    if assigned_user == "No eligible users found":
        st.error("No eligible users available for this task based on skills.")
    else:
        # Update user load count dynamically
        st.session_state.user_load[assigned_user] += 1

        # Save to history
        st.session_state.history.append({
            "Task": task_input,
            "Predicted Type": predicted_type,
            "Priority": predicted_priority,
            "Assigned To": assigned_user
        })

        st.success(f"✅ Task assigned to: **{assigned_user}**")
        st.info(f"⚡ Priority: **{predicted_priority}**")
        st.caption(f"🗂️ Type: **{predicted_type}**")

# Show Task History
if st.session_state.history:
    st.markdown("---")
    st.subheader("📜 Task History")
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df, use_container_width=True)
