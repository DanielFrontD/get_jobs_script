import re
import nltk
import pandas as pd
import docx
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

jobs_file = pd.read_csv("cvs_dataset.csv", encoding='cp1252')
desc = jobs_file['DESCRIPCION']

def clear_data(text):
    onlyLetters = re.sub("[^a-zA-ZáéíóúñÑ]", " ", text)
    words = onlyLetters.lower().split()
    stops = set(stopwords.words('spanish'))
    meaningful_words = [w for w in words if not w in stops]
    return " ".join(meaningful_words)

def getText(filename):
    doc = docx.Document(filename)
    fullText = []
    for para in doc.paragraphs:
        fullText.append(para.text)
    return '\n'.join(fullText)

print('GETTING DATASET')
num_filas = desc.size
desc_limpio = []
for i in range(0, num_filas):
    desc_limpio.append(clear_data(desc[i]))
print('DATASET -> COMPLETE')

desc_limpio = pd.Series(desc_limpio)

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(desc_limpio)

print('GETTING CV')
read_word = getText('CV_Abogado.docx')
cv_doc_cleared = clear_data(read_word)
pd_cv = pd.Series(cv_doc_cleared)

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_matrix2 = tfidf_vectorizer.transform(pd_cv)
print('CV -> COMPLETE')

res = cosine_similarity(tfidf_matrix2, tfidf_matrix, True)
res = sorted(res[0], reverse=True)

res = cosine_similarity(tfidf_matrix2, tfidf_matrix, True)
res = res[0]

size = len(res)
job_simil = pd.DataFrame(columns=('ID', 'Puesto', 'Similitud'))
i = int()
for i in range(0, size):
    job_simil.loc[i] = [i+1, jobs_file['PUESTO'][i],res[i]]

sorted_job = job_simil.sort_values(['Similitud'], ascending=False)

print('LOOKING FOR JOBS')
export_jobs = pd.DataFrame(sorted_job, columns=('ID', 'Puesto', 'Similitud'))
export_jobs.to_csv(r'./jobs.csv', index = None, header=True)
print('BEST JOBS EXPORTED')