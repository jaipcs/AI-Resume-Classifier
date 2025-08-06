import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
from collections import Counter
from docx import Document
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import zipfile
import io
import os
import pdfplumber
from sklearn.cluster import KMeans
import shap

# ----------------- Setup -----------------

nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ----------------- Load Model Artifacts -----------------
model = joblib.load("best_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")
le = joblib.load("label_encoder.pkl")

# ----------------- Constants -----------------

TECH_SKILLS = [
    "python","sql","machine learning","data analysis","tensorflow","keras","pytorch",
    "streamlit","sklearn","pandas","numpy","seaborn","matplotlib","deep learning",
    "nlp","cloud","azure","aws","gcp","c++","java","powerbi","excel","hadoop"
]

ROLE_REQUIREMENTS = {
    "Data Scientist": ["python","machine learning","pandas","numpy","deep learning"],
    "AI Engineer": ["python","pytorch","tensorflow","nlp","deep learning"],
    "Data Analyst": ["sql","excel","powerbi","data analysis","pandas"],
    "ML Engineer": ["python","scikit-learn","tensorflow","machine learning"],
    "Machine Learning Engineer": ["ml engineer", "deep learning", "neural network", "pytorch", "tensorflow", "computer vision"],
    "SQL Developer": ["sql", "database", "pl/sql", "stored procedure", "oracle", "mysql", "postgresql", "dbms"],
    "React Developer": ["react", "javascript", "frontend", "web developer", "redux", "html", "css", "typescript"],
    "Java Developer": ["java", "spring", "hibernate", "backend", "microservices", "j2ee", "servlet"],
    "AWS Engineer": ["aws", "cloud", "ec2", "lambda", "s3", "cloudformation", "cloudwatch", "devops"],
    "DevOps Engineer": ["devops", "ci/cd", "jenkins", "docker", "kubernetes", "ansible", "terraform", "automation"],
    "Full Stack Developer": ["full stack", "mern", "mean", "node", "react", "angular", "express", "mongodb", "javascript"],
    "Python Developer": ["python", "django", "flask", "rest api", "fastapi"],
    "Mobile App Developer": ["android", "ios", "react native", "flutter", "kotlin", "swift"],
    "Software Tester": ["testing", "qa", "selenium", "manual testing", "automation testing", "test case", "bug tracking"],
    "Data Engineer": ["etl", "hadoop", "spark", "big data", "data pipeline", "airflow", "kafka"],
    "Cloud Engineer": ["azure", "gcp", "cloud", "iac", "serverless", "kubernetes"],
    "Business Analyst": ["business analyst", "requirements", "stakeholder", "gap analysis", "process mapping"],
    "Project Manager": ["project manager", "scrum master", "agile", "pmp", "stakeholder management"]
}

# ----------------- Helper Functions -----------------

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return " ".join(tokens)

def extract_text_from_docx(file):
    doc = Document(file)
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def extract_text_from_resume(file):
    name = file.name.lower()
    if name.endswith(".docx"):
        return extract_text_from_docx(file)
    elif name.endswith(".pdf"):
        return extract_text_from_pdf(file)
    return ""

def extract_skills(text):
    return [s for s in TECH_SKILLS if s in text.lower()]

def extract_experience(text):
    years = re.findall(r'(\d+)\s*(?:\+?\s*years?|yrs?|year)', text.lower())
    return max(map(int, years), default=0)

def get_role_fit(skills, predicted_role):
    required = ROLE_REQUIREMENTS.get(predicted_role, [])
    matched = len(set(skills) & set(required))
    return round((matched / len(required) * 100), 1) if required else 50

def send_email(to_email, candidate_name, predicted_role):
    sender_email = "your_email@gmail.com"  # Replace
    password = "your_app_password"         # Replace

    subject = f"Interview Invitation for {predicted_role}"
    body = f"""
    Dear {candidate_name},

    We reviewed your resume and found your skills suitable for the {predicted_role} role.
    We would like to invite you for an interview.

    Regards,
    HR Team
    """

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, password)
            server.sendmail(sender_email, to_email, msg.as_string())
        return True
    except Exception:
        return False

def process_uploaded_files(uploaded_files):
    resumes = []
    for file in uploaded_files:
        if file.name.endswith(".zip"):
            with zipfile.ZipFile(file) as z:
                for name in z.namelist():
                    if name.endswith((".docx", ".pdf")):
                        with z.open(name) as extracted:
                            text = extract_text_from_resume(io.BytesIO(extracted.read()))
                            resumes.append((name, text))
        else:
            text = extract_text_from_resume(file)
            resumes.append((file.name, text))
    return resumes

# ----------------- Streamlit Layout -----------------
st.set_page_config(page_title="AI Recruitment Intelligence Dashboard", layout="wide")
st.title("AI-Powered Resume Classifier & Analytics (Pro+)")

uploaded_files = st.file_uploader("Upload Resumes (DOCX/PDF/ZIP)", type=["docx","pdf","zip"], accept_multiple_files=True)

if uploaded_files:
    # Process resumes
    raw_resumes = process_uploaded_files(uploaded_files)

    results = []
    all_skills = []
    role_counts = Counter()

    for name, text in raw_resumes:
        cleaned = clean_text(text)
        pred_probs = model.predict_proba(tfidf.transform([cleaned]))[0]
        top_roles_idx = np.argsort(pred_probs)[::-1][:3]
        top_roles = [(le.inverse_transform([i])[0], round(pred_probs[i]*100,2)) for i in top_roles_idx]

        predicted_role = top_roles[0][0]
        confidence = top_roles[0][1]

        skills = extract_skills(text)
        exp_years = extract_experience(text)
        role_fit_score = get_role_fit(skills, predicted_role)

        results.append({
            "Candidate": name,
            "Skills": ", ".join(skills),
            "Experience (Years)": exp_years,
            "Predicted Role": predicted_role,
            "Confidence (%)": confidence,
            "Role Fit (%)": role_fit_score,
            "Top 3 Roles": str(top_roles)
        })
        all_skills.extend(skills)
        role_counts[predicted_role] += 1

    df_results = pd.DataFrame(results)

    # Tabs: 8 modules + advanced analytics
    tabs = st.tabs([
        "Resume Insights", "Visualizations", "Interactive Tools", "Bulk Email",
        "Prediction Enhancements", "Multi-Format Support", "UI/UX Enhancements",
        "Analytics Dashboard", "Advanced Analytics"
    ])

    # 1. Resume Insights
    with tabs[0]:
        st.subheader("Candidate Insights")
        st.dataframe(df_results)

        # Skill Gaps
        st.subheader("Skill Gaps vs Role Requirements")
        for idx, row in df_results.iterrows():
            required = set(ROLE_REQUIREMENTS.get(row["Predicted Role"], []))
            candidate_skills = set(row["Skills"].split(", "))
            gaps = required - candidate_skills
            st.write(f"**{row['Candidate']}** → Missing Skills: {', '.join(gaps) if gaps else 'None'}")

    # 2. Visualizations
    with tabs[1]:
        st.subheader("Skills Frequency")
        skill_freq = Counter(all_skills)
        if skill_freq:
            skill_df = pd.DataFrame(skill_freq.items(), columns=["Skill", "Count"]).sort_values(by="Count", ascending=False)
            st.bar_chart(skill_df.set_index("Skill"))

        st.subheader("Experience Distribution")
        st.bar_chart(df_results.set_index("Candidate")["Experience (Years)"])

        st.subheader("Predicted Roles Distribution")
        role_df = pd.DataFrame(role_counts.items(), columns=["Role", "Count"])
        st.bar_chart(role_df.set_index("Role"))

    # 3. Interactive Tools
    with tabs[2]:
        st.subheader("Filter Candidates")
        min_exp = st.slider("Minimum Experience", 0, 20, 0)
        role_filter = st.selectbox("Filter by Role", ["All"] + df_results["Predicted Role"].unique().tolist())
        filtered = df_results[df_results["Experience (Years)"] >= min_exp]
        if role_filter != "All":
            filtered = filtered[filtered["Predicted Role"] == role_filter]
        st.dataframe(filtered)

        st.subheader("Shortlist & Export")
        shortlisted = st.multiselect("Select Candidates", df_results["Candidate"].tolist())
        if shortlisted:
            st.download_button("Download Shortlisted CSV", df_results[df_results["Candidate"].isin(shortlisted)].to_csv(index=False), "shortlisted_candidates.csv")

    # 4. Bulk Email
    with tabs[3]:
        st.subheader("Bulk Email Shortlisted Candidates")
        email_list = st.text_area("Enter Emails (comma separated)")
        names_list = st.text_area("Enter Candidate Names (comma separated)")
        if st.button("Send Emails"):
            emails = [e.strip() for e in email_list.split(",") if e.strip()]
            names = [n.strip() for n in names_list.split(",") if n.strip()]
            for email, name in zip(emails, names):
                send_email(email, name, "Predicted Role")
            st.success("Emails sent successfully (check SMTP config).")

    # 5. Prediction Enhancements
    with tabs[4]:
        st.subheader("Confidence Scores & Top 3 Roles")
        st.dataframe(df_results[["Candidate", "Predicted Role", "Confidence (%)", "Top 3 Roles"]])

    # 6. Multi-Format Support
    with tabs[5]:
        st.info("DOCX, PDF, and ZIP uploads are already supported.")

    # 7. UI/UX Enhancements
    with tabs[6]:
        st.info("Dark mode, PDF reports, and advanced navigation can be added here.")

    # 8. Analytics Dashboard
    with tabs[7]:
        st.subheader("Summary Analytics")
        st.write(f"Total Candidates: {len(df_results)}")
        st.write(f"Average Experience: {df_results['Experience (Years)'].mean():.2f} years")
        st.write("Role Distribution Table:")
        st.dataframe(role_df)

    # 9. Advanced Analytics
    with tabs[8]:
        st.subheader("Candidate Clustering (Skill-Based)")
        # Convert skills to vector for clustering
        skill_matrix = pd.DataFrame(0, index=df_results["Candidate"], columns=TECH_SKILLS)
        for idx, skills in enumerate(df_results["Skills"]):
            for s in skills.split(", "):
                if s in TECH_SKILLS:
                    skill_matrix.loc[df_results["Candidate"][idx], s] = 1
        if len(skill_matrix) > 1:
            kmeans = KMeans(n_clusters=min(3, len(skill_matrix)), random_state=42).fit(skill_matrix)
            df_results["Cluster"] = kmeans.labels_
            st.dataframe(df_results[["Candidate","Predicted Role","Cluster"]])

        st.subheader("Explainable AI (Feature Importance with SHAP)")
        try:
            explainer = shap.Explainer(model, tfidf.transform([clean_text(" ".join(all_skills))]))
            shap_values = explainer(tfidf.transform([clean_text(" ".join(all_skills))]))
            st.write("Top features influencing predictions shown below:")
            st.pyplot(shap.plots.bar(shap_values))
        except Exception:
            st.info("SHAP visualization not available for this model type.")

    # Download all results
    st.download_button("Download Full Candidate Data", df_results.to_csv(index=False), "all_candidates.csv")
  
