import pandas as pd

df = pd.read_csv("./data/course_061120.csv")
print(df.head(50))

# TODO: Return top 5 recommendations with confidence level > 0.3

# Import TfIdfVectorizer from the scikit-learn library
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Define a TF-IDF Vectorizer Object. Remove all english stopwords
tfidf = TfidfVectorizer(stop_words='english')

# Replace NaN with an empty string
df['Description'] = df['Description'].fillna('')

# Construct the required TF-IDF matrix by applying the fit_transform method on the overview feature
tfidf_matrix = tfidf.fit_transform(df['Description'])

# Output the shape of tfidf_matrix
print(tfidf_matrix.shape)

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
d = df.drop_duplicates(subset=['Title'])

indices = pd.Series(d.index, index=d['Title'])


# Function that takes in movie title as input and gives recommendations
def content_recommender(title, cosine_sim=cosine_sim, df=df, indices=indices):
    # Obtain the index of the movie that matches the title
    idx = indices[title]
    print(idx)

    # Get the pairwsie similarity scores of all movies with that movie
    # And convert it into a list of tuples as described above
    sim_scores = list(enumerate(cosine_sim[idx]))

    # TODO: Pick idx item of the second item of the tuple, this is the similarity between the 2 courses
    # TODO: return items where score > 0.3
    sort = []
    for item in sim_scores:
        if item[1] > 0.3:
            sort.append(item)
    # Sort the movies based on the cosine similarity scores
    sort = sorted(sort, key=lambda x: x[1], reverse=True)
    # Get the scores of the 5 most similar courses. Ignore the first course.
    sim_scores = sort[1:6]
    print(sim_scores)
    print([])
    indxs = []
    for item in sim_scores:
        indxs.append(item[0])

    # Return the top 10 most similar movies
    return d['Title'].iloc[indxs]


print(content_recommender('Emotional Intelligence'))

# TODO: Group object in Clusters (UI?)

# TODO: Plot objects per cluster (UI)

# TODO: Create object locator (UI) (search bar)

# TODO: Zoom in on object (UI)
