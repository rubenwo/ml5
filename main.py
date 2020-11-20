import pandas as pd

df = pd.read_csv("./data/course_061120.csv")
print(df.head(50))

# TODO: Return top 5 recommendations with confidence level > 0.3

# Import TfIdfVectorizer from the scikit-learn library
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

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
kmeans_per_k = [MiniBatchKMeans(n_clusters=k, random_state=42).fit(cosine_sim)
                for k in range(1, 25)]
WCSS = [model.inertia_ for model in kmeans_per_k]

print(WCSS)

plt.figure()
plt.plot(range(1, 25), WCSS)
plt.xlabel("$k$", fontsize=14)
plt.ylabel("WCSS", fontsize=14)
plt.annotate('Elbow',
             xy=(4, WCSS[3]),
             xytext=(0.55, 0.55),
             textcoords='figure fraction',
             fontsize=16,
             arrowprops=dict(facecolor='black', shrink=0.1)
             )
plt.axis([1, 25, 0, 100000])
plt.show()

smaller_df = TSNE().fit_transform(PCA(n_components=2).fit_transform(cosine_sim))
print(smaller_df)

k = 4
kmeans = MiniBatchKMeans(n_clusters=k, random_state=42)
y_pred = kmeans.fit_predict(cosine_sim)


def plot_clusters(X, y=None):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='viridis')
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()


plot_clusters(smaller_df, y=y_pred)


# TODO: Plot objects per cluster (UI)

# TODO: Create object locator (UI) (search bar)
def get_coordinates(X, plot_data, title):
    # Obtain the index of the movie that matches the title
    idx = X[title]
    return plot_data[idx] if (idx is not None) else [0, 0]


def plot_clusters_with_target(X, target, y=None, zoom_axis=0):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='viridis')
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.scatter(target[0], target[1], color="red")
    plt.annotate("Learning Object", (target[0], target[1]))
    plt.tight_layout()
    if zoom_axis != 0:  # Zoom in on object (UI)
        plt.axis([target[0]-zoom_axis, target[0]+zoom_axis, target[1]-zoom_axis, target[1]+zoom_axis])
    plt.show()


target_point = get_coordinates(indices, smaller_df, 'Eager to help? ~ Share your expertise (WIP)')
plot_clusters_with_target(smaller_df, target_point, y_pred)
plot_clusters_with_target(smaller_df, target_point, y_pred, zoom_axis=30)



