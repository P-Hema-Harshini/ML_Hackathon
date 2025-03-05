import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import requests
from deep_translator import GoogleTranslator
import langid
import re

headers = {
    "X-RapidAPI-Key": "86ab80ae35msh8e76e31eeb1ad7ap13511bjsn0d7e56185828",  # Use Streamlit secrets
    "x-rapidapi-host": "linkedin-data-api.p.rapidapi.com"
}

@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        spacy.cli.download("en_core_web_sm")
        return spacy.load("en_core_web_sm")

nlp = load_spacy_model()

def load_user_data(profile_url):
    x = "https://linkedin-data-api.p.rapidapi.com/get-profile-data-by-url?url="
    url = str(x + profile_url)
    querystring = {"s": "Inception", "r": "json", "page": "2"}
    response = requests.get(url, headers=headers, params=querystring)
    if response.status_code == 200:
        data3 = response.json()
        skills = [skill["name"] for skill in data3.get("skills", [])]
        return skills
    else:
        st.error(f"API Error: {response.status_code}")
        return []

def load_job_skills(jobn):
    x = "https://linkedin-data-api.p.rapidapi.com/search-jobs?keywords="
    y = "&datePosted=anyTime&sort=mostRelevant"
    url = str(x + jobn + y)
    querystring = {"s": "Inception", "r": "json", "page": "2"}
    response = requests.get(url, headers=headers, params=querystring)
    if response.status_code == 200:
        dataloc = response.json()
        if dataloc.get("data"):
            job_id = dataloc["data"][0]["id"] #get the first job returned.
            datagot = load_job_data(job_id)
            if datagot and datagot.get("data") and datagot["data"].get("description"):
                data_of_job = datagot["data"]["description"]
                translator = GoogleTranslator(source="auto", target="en")
                def translate_to_english(text):
                    try:
                        detected_lang = langid.classify(text)[0]
                        if detected_lang != "en":
                            return translator.translate(text)
                        return text
                    except Exception as e:
                        st.error(f"Translation failed: {e}")
                        return text
                data_of_job = translate_to_english(data_of_job)
                pattern = r"(?:Requirements|Qualifications|Skills needed|Experience)[:\s]*(.*?)(?=\n[A-Z]|\Z)"
                matches = re.findall(pattern, data_of_job, re.DOTALL)
                return matches if matches else ["No requirements found"]
            else:
                st.error("Job details not found.")
                return []
        else:
            st.error("No jobs found for that search.")
            return []
    else:
        st.error(f"API Error: {response.status_code}")
        return []

def load_job_data(id):
    r = "https://linkedin-data-api.p.rapidapi.com/get-job-details?id="
    url = str(r + id)
    querystring = {"s": "Inception", "r": "json", "page": "2"}
    response = requests.get(url, headers=headers, params=querystring)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"API Error: {response.status_code}")
        return None

def suggest_missing_skills(user_skills, job_skills, vectorizer, top_n=5):
    user_skills_str = " ".join(user_skills)
    user_skills_vector = vectorizer.transform([user_skills_str])
    skill_similarities = []
    for job_skill in job_skills:
        job_skill_vector = vectorizer.transform([job_skill])
        similarity = cosine_similarity(job_skill_vector, user_skills_vector)[0][0]
        skill_similarities.append((job_skill, similarity))
    skill_similarities.sort(key=lambda x: x[1])
    return [skill[0] for skill in skill_similarities[:top_n]]

def job_defined(skills):
    doc = nlp(skills)
    potential_skills = [chunk.text for chunk in doc.noun_chunks]
    return potential_skills

st.title("Skill Gap Analyzer")

user_profile = st.text_area("Enter your Linkedin profile URL")
job_title = st.text_input("Enter the job title you want", placeholder="Data Scientist")

if st.button("Predict Missing Skills"):
    with st.spinner("Loading data..."):
        user_skills = load_user_data(user_profile)
        job_skills = load_job_skills(job_title)
