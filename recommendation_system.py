import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from scipy.sparse import hstack

# Load job opportunities data
file_path = 'data/PreProcessedOpportunities.csv'
opp_df = pd.read_csv(file_path)

# Columns of interest for job opportunities
job_columns = ['Description', 'Skills', 'Majors', 'Location', 'GPA out of 5', 'GPA out of 4', 'Job Title']

# Ensure each field is treated as a string
for column in job_columns:
    opp_df[column] = opp_df[column].fillna('').astype(str)

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


skills = [
    "Adobe XD", "Agile", "Angular", "API integration (REST)", "API integration (SOAP)", "ASP.NET", "AWS", 
    "Azure", "Big Data Analytics", "Bitbucket", "Blockchain", "C#", "C++", "Cloud Architecture", "Confluence", 
    "CRM systems", "CSS", "Cybersecurity", "Data Analysis", "Data Mining", "Data Visualization", "Database Design",
    "DevOps", "Docker", "Encryption", "Excel", "Express", "Figma", "Firebase", "Financial Forecasting",
    "Financial Modeling", "Firewalls", "Git and GitHub", "GCP", "Google Ads", "Google Analytics", "Hadoop", 
    "HTML", "Investment Analysis", "Java", "JavaScript", "Jest", "JIRA", "JUnit", "Kubernetes", "Machine Learning", 
    "MATLAB", "Microsoft Office Suite", "MongoDB", "MS Project", "Network Fundamentals", "NoSQL", "Node.js", 
    "NLP", "Object-Oriented Programming (OOP)", "Oracle APEX", "Penetration Testing", "PHP", "PL/SQL", "Postman", 
    "Power BI", "PowerPoint", "Prototyping", "Python", "R Programming", "React", "Ruby", "Selenium", "Sketch", 
    "SQL", "Statistical Analysis", "Supervised/Unsupervised Learning", "SVN", "Swift", "Tableau", "TensorFlow", 
    "Trello", "UI/UX Design", "User Research", "VLOOKUP", "Vue.js", "Waterfall", "Web Development", "Word"
]

 # Convert all skills to lowercase
skills = [skill.lower() for skill in skills]

def vectorize_opp(opp_df):
    # TF-IDF vectorization for 'Description', 'Skills', 'Job Title', and 'Majors'
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    opp_text_vectors = tfidf_vectorizer.fit_transform(
        opp_df['Description'] + ' ' + opp_df['Skills'] + ' ' + opp_df['Job Title'] + ' ' + opp_df['Majors']
    )

    # Multi-hot encoding for 'Location'
    location_encoder = MultiLabelBinarizer(classes=cities)
    opp_location_vectors = location_encoder.fit_transform(opp_df['Location'])


    # GPA normalization
    gpa_scaler = StandardScaler()
    # Convert GPA columns to numeric, coerce errors to NaN, then fill NaN with 0
    gpa_out_of_5 = pd.to_numeric(opp_df['GPA out of 5'], errors='coerce').fillna(0)
    gpa_out_of_4 = pd.to_numeric(opp_df['GPA out of 4'], errors='coerce').fillna(0)
    opp_gpa_vectors = gpa_scaler.fit_transform(pd.DataFrame({'GPA out of 5': gpa_out_of_5, 'GPA out of 4': gpa_out_of_4}))

    # Combine all features into a single sparse matrix
    opp_vectors = hstack([opp_text_vectors, opp_location_vectors, opp_gpa_vectors])
    return opp_vectors

# Define a function to vectorize the user profile
def vectorize_user(user_data):
    # Convert user's major and skills to lowercase for consistency
    user_data['major'] = user_data['major'].lower()
    user_data['skills'] = [skill.lower() for skill in user_data['skills']]

    # TF-IDF vectorization for 'major' and 'skills' with lowercase=False
    tfidf_vectorizer = TfidfVectorizer(vocabulary=skills + [user_data['major']], stop_words='english', lowercase=False)
    user_text_vector = tfidf_vectorizer.fit_transform([' '.join([user_data['major']] + user_data['skills'])])

    # Multi-hot encode the 'location' (multiple selection)
    location_encoder = MultiLabelBinarizer(classes=cities)
    user_location_vector = location_encoder.fit_transform([user_data['location']])

    # Normalize GPA values
    scaler = StandardScaler()
    user_gpa_vector = scaler.fit_transform([[user_data['gpa_scale'], user_data['gpa']]])

    # Combine all user features into a single sparse matrix
    user_vector = hstack([user_text_vector, user_location_vector, user_gpa_vector])
    return user_vector

