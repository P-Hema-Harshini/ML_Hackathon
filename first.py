import streamlit as st
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import spacy
import subprocess

# Check if "en_core_web_sm" is installed, else download it
# try:
#     nlp = spacy.load("en_core_web_sm")
# except OSError:
#     subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
#     nlp = spacy.load("en_core_web_sm")  # Load after downloading

def load_user_data(profile_url):
    import requests
    x="https://linkedin-data-api.p.rapidapi.com/get-profile-data-by-url?url="
    url = str(x+profile_url)  # Actual API URL
    querystring = {"s": "Inception", "r": "json", "page": "2"}  # Adjust based on API docs

    headers = {
        "X-RapidAPI-Key": "4f22906f26mshe5d92a6ef0415bfp1d976fjsn4ee2fba12f7e",  # Replace with your API key
        "x-rapidapi-host": "linkedin-data-api.p.rapidapi.com"
    }
    response = requests.get(url,headers=headers,params=querystring)
    try:
            data3 = response.json()
            # return data3
    except requests.exceptions.JSONDecodeError:
            print("Error: Response is not in JSON format. Raw response:")
            print(response.text)
    # full_name = f"{data3['firstName']} {data3['lastName']}"

    # Extract skills as a list
    skills = [skill["name"] for skill in data3.get("skills",[])]
    # st.write(skills)
    df=pd.DataFrame()
    # Convert to DataFrame
    dff = pd.DataFrame({"Skills": [", ".join(skills)]})  # Store skills as a single string
    # df= pd.concat([df, dff], ignore_index=True)
    # Display DataFrame
    # st.write(dff)
    return dff
def load_job_skills(jobn):
    import requests
    x="https://linkedin-data-api.p.rapidapi.com/search-jobs?keywords="
    y="&datePosted=anyTime&sort=mostRelevant"
    url = str(x+jobn+y)  # Actual API URL
    querystring = {"s": "Inception", "r": "json", "page": "2"}  # Adjust based on API docs

    headers = {
        "X-RapidAPI-Key": "4f22906f26mshe5d92a6ef0415bfp1d976fjsn4ee2fba12f7e",  # Replace with your API key
        "x-rapidapi-host": "linkedin-data-api.p.rapidapi.com"
    }
    response = requests.get(url,headers=headers,params=querystring)
    try:
            dataloc = response.json()
            # print(dataloc)
    except requests.exceptions.JSONDecodeError:
            print("Error: Response is not in JSON format. Raw response:")
            print(response.text)
    # st.write(dataloc)        
    job_id=dataloc['data'][2]['id']
    datagot=load_job_data(job_id)
    data_of_job=datagot['data']['description']
    from deep_translator import GoogleTranslator
    import langid  # More reliable than langdetect

    translator = GoogleTranslator(source="auto", target="en")  # Auto-detect source language

    def translate_to_english(text):
        try:
            detected_lang = langid.classify(text)[0]  # Detect language
            if detected_lang != "en":  # If not English, translate
                translated_text = translator.translate(text)  # No need for src/dest
                return translated_text
            return text  # If already English, return as is
        except Exception as e:
            print(f"Translation failed: {e}")
            return text  

    data_of_job=translate_to_english(data_of_job) # Output: "Hello, how are you?"
    import re

    def extract_requirements(description):
        # Define regex pattern to extract requirements
        pattern = r"(?:Requirements|Qualifications|Skills needed|Experience)[:\s]*(.*?)(?=\n[A-Z]|\Z)"
        matches = re.findall(pattern,description, re.DOTALL)
        return matches if matches else "No requirements found"
    skills_he_need=extract_requirements(data_of_job)
    return skills_he_need
def load_job_data(id):
      
    import requests
    r="https://linkedin-data-api.p.rapidapi.com/get-job-details?id="
    url = str(r+id)# Actual API URL
    querystring = {"s": "Inception", "r": "json", "page": "2"}  # Adjust based on API docs

    headers = {
        "X-RapidAPI-Key": "4f22906f26mshe5d92a6ef0415bfp1d976fjsn4ee2fba12f7e",  # Replace with your API key
        "x-rapidapi-host": "linkedin-data-api.p.rapidapi.com"
    }
    response = requests.get(url,headers=headers,params=querystring)
    try:
            dataj = response.json()
            return dataj
    except requests.exceptions.JSONDecodeError:
            print("Error: Response is not in JSON format. Raw response:")
            print(response.text)
        
# Compare user skills with job requirements
def suggest_missing_skills(jobs_he_has, skills_he_need, vectorizer, top_n=5):
    user_skills_str = " ".join(jobs_he_has)  # Convert user skills to a single string
    user_skills_vector = vectorizer.transform([user_skills_str])  # Vectorize user skills

    skill_similarities = []

    for job_skill in skills_he_need:
        job_skill_vector = vectorizer.transform([job_skill])  # ✅ Transform individual skill
        similarity = cosine_similarity(job_skill_vector, user_skills_vector)[0][0]  # Compute similarity
        skill_similarities.append((job_skill, similarity))

    # Sort by similarity (lowest first)
    skill_similarities.sort(key=lambda x: x[1])

    # Return the top N missing skills
    return [skill[0] for skill in skill_similarities[:top_n]]


def job_defined(skills):
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(skills)
    potential_skills = []
    for chunk in doc.noun_chunks:
        potential_skills.append(chunk.text)
    return potential_skills

# Streamlit UI
st.title("🔍 Skill Gap Analyzer")

# User Inputs
user_profile = st.text_area("📝 Enter your Linkedin profile URL")
job_title = st.text_input("💼 Enter the job title you want", placeholder="Data Scientist")

# Button to trigger prediction
if st.button("🔍 Predict Missing Skills"):
    user_skills = load_user_data(user_profile)
    job_skills=load_job_skills(job_title)
    job_redefined = job_defined(" ".join(job_skills))  # Convert list to string
    all_skills=[]
    all_skills.extend([skill.strip() for skill in user_skills])
    all_skills.extend([req.strip() for req in job_redefined])
    vectorizers = TfidfVectorizer()
    vectorizers.fit(all_skills)
    suggested_skills=suggest_missing_skills(user_skills,job_skills, vectorizers)
    st.write(suggested_skills) 
