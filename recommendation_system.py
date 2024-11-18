import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from scipy.sparse import hstack, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import firebase_admin
from firebase_admin import credentials, firestore
import numpy as np
import re

# Initialize Firebase Admin SDK
cred = credentials.Certificate("C:/Users/aljaw/Desktop/GP1/hadafi-a7971-firebase-adminsdk-3k84a-901d21ad25.json")
firebase_admin.initialize_app(cred)

# Access Firestore database
db = firestore.client()

# Load job opportunities data
file_path = 'C:/Users/aljaw/Desktop/GP1/preProcessing.csv'
opp_df = pd.read_csv(file_path)

# Columns of interest for job opportunities
job_columns = ['Description', 'Skills', 'Majors', 'Location', 'GPA out of 5', 'GPA out of 4', 'Job Title']

# Ensure each field is treated as a string
for column in job_columns:
    opp_df[column] = opp_df[column].fillna('').astype(str)


# Convert GPA columns to numeric, replacing NaN with 0
opp_df['GPA out of 5'] = pd.to_numeric(opp_df['GPA out of 5'], errors='coerce').fillna(0)
opp_df['GPA out of 4'] = pd.to_numeric(opp_df['GPA out of 4'], errors='coerce').fillna(0)

# Initialize lists for selection fields (for encoding consistency)
cities = [
    'Abha', 'Al Ahsa', 'Al Khobar', 'Al Qassim', 'Dammam', 'Hail', 'Jeddah', 'Jizan', 'Jubail',
    'Mecca', 'Medina', 'Najran', 'Riyadh', 'Tabuk', 'Taif'
]

# Standardize any alternate spellings
opp_df['Location'] = opp_df['Location'].str.replace('Jiddah', 'Jeddah')

# Function to expand "Saudi Arabia" to all cities
def expand_saudi_arabia(locations):
    if 'Saudi Arabia' in locations:
        return cities  # Replace "Saudi Arabia" with the list of all cities
    return locations

# Apply the function to each row
opp_df['Location'] = opp_df['Location'].apply(lambda x: expand_saudi_arabia(x.split(',')))


# Initialize location binarizer
location_binarizer = MultiLabelBinarizer()
location_binarizer.fit(opp_df['Location'])

# Shared TfidfVectorizer: skills 
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
tfidf_vectorizer.fit(opp_df['Skills'])

# Shared StandardScaler for GPA
gpa_scaler = StandardScaler()
gpa_scaler.fit(opp_df[['GPA out of 5', 'GPA out of 4']])

# Modular function to vectorize opportunities
def vectorize_opp():
    opp_text_vectors = tfidf_vectorizer.transform(opp_df['Skills'])
    opp_gpa_vectors = csr_matrix(gpa_scaler.transform(opp_df[['GPA out of 5', 'GPA out of 4']]))
    opp_location_vectors = csr_matrix(location_binarizer.transform(opp_df['Location']))
    return opp_text_vectors, opp_gpa_vectors, opp_location_vectors

# Call vectorize_opportunities and store results
opp_text_vectors, opp_gpa_vectors, opp_location_vectors = vectorize_opp()

# Modular function to vectorize a user
def vectorize_user(user_data):
    user_data['skills'] = [skill.lower() for skill in user_data['skills']]

    user_text_vector = tfidf_vectorizer.transform([' '.join(user_data['skills'])])
    
    # Fix for StandardScaler input
    user_gpa_df = pd.DataFrame(
        [[user_data['gpa'], user_data['gpaScale']]],
        columns=['GPA out of 5', 'GPA out of 4']  # Same column names as used during fitting
    )
    user_gpa_vector = csr_matrix(gpa_scaler.transform(user_gpa_df))
    
    user_location_vector = csr_matrix(location_binarizer.transform([user_data['location']]))

    return user_text_vector, user_gpa_vector, user_location_vector


# Similarity Calculation: GPA
def calculate_gpa_similarity(user_gpa, user_gpa_scale, job_gpa_5, job_gpa_4):
    if job_gpa_5 == 0 and job_gpa_4 == 0:
        return 1.0

    # Scale user GPA to match the job GPA scale
    if user_gpa_scale == 5:
        user_gpa_scaled = user_gpa
    elif user_gpa_scale == 4:
        user_gpa_scaled = user_gpa * (5 / 4)
    else:
        raise ValueError("Unsupported GPA scale. Only 4 or 5 are allowed.")
    
    # Get the job's required GPA in the same scale (out of 5)
    job_gpa = max(job_gpa_5, job_gpa_4 * (5 / 4))
    
    # If the user GPA matches or is higher than required: Full score
    if user_gpa_scaled >= job_gpa:
        return 1.0
    
    # if lower, calculate the similarity as a function of the difference
    return 1 - abs(user_gpa_scaled - job_gpa) / 5

# Similarity Calculation: Skills: If user is overqualified: full score. Otherwise each skill has its weight
def calculate_skills_similarity(user_skills_vector, job_skills_vector):
    return cosine_similarity(user_skills_vector, job_skills_vector)[0][0]

# Similarity Calculation: Location: If one matches: Full score.
def calculate_location_similarity(user_locations, job_locations):
    
    user_locations_array = user_locations.toarray().flatten()
    job_locations_array = job_locations.toarray().flatten()

    # Check if there is any matches
    if np.dot(user_locations_array, job_locations_array) > 0:
        return 1.0  # At least one location matches
    return 0.0  # No matches

# Fitch all users from database
def fetch_users_from_firestore():
    student_ref = db.collection("Student")
    return [doc.to_dict() for doc in student_ref.stream()]

# Combine similatites and print recommendations
def generate_recommendations():
    students = fetch_users_from_firestore()
    for student in students:
        try:
            user_text_vector, user_gpa_vector, user_location_vector = vectorize_user(student)
            
            recommendations = []
            for i, row in opp_df.iterrows():
                if student['major'].lower() in map(str.strip, row['Majors'].lower().split(',')):
                    job_text_vector = opp_text_vectors[i]
                    job_gpa_vector = opp_gpa_vectors[i]
                    job_location_vector = opp_location_vectors[i]

                    skills_similarity = calculate_skills_similarity(user_text_vector, job_text_vector)
                    location_similarity = calculate_location_similarity(user_location_vector, csr_matrix(job_location_vector))
                    gpa_similarity = calculate_gpa_similarity(
                        user_gpa=float(student['gpa']),
                        user_gpa_scale=float(student['gpaScale']),
                        job_gpa_5=row['GPA out of 5'],
                        job_gpa_4=row['GPA out of 4']
                    )
                    
                    total_similarity = 0.34 * skills_similarity + 0.33 * location_similarity + 0.33 * gpa_similarity

                    recommendations.append({
                        'Job Title': row['Job Title'],
                        'Description': row['Description'],
                        'Job Skills': row['Skills'],
                        'Skills Similarity': skills_similarity,
                        'Location Similarity': location_similarity,
                        'GPA Similarity': gpa_similarity,
                        'Total Similarity': total_similarity
                    })

            recommendations = sorted(recommendations, key=lambda x: x['Total Similarity'], reverse=True)

            print(f"\nRecommendations for user UID ({student['uid']}):")
            print(f"User Skills: {', '.join(student['skills'])}")
            for idx, job in enumerate(recommendations, start=1):
                print(f"{idx}. Job Title: {job['Job Title']}")
                print(f"   Description: {job['Description']}")
                print(f"   Total Similarity: {job['Total Similarity']:.2f}\n")
        except Exception as e:
            print(f"Error generating recommendations for user UID ({student['uid']}): {e}")

if __name__ == "__main__":
    generate_recommendations()

