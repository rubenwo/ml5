import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import linear_kernel

df = pd.read_csv("./data/course_061120.csv")
print(df.head(50))

# Define a TF-IDF Vectorizer Object. Remove all english stopwords
tfidf = TfidfVectorizer(stop_words='english')

# Replace NaN with an empty string
df['Description'] = df['Description'].fillna('')

# Create the tf-idf matrix using the description from the dataset
tfidf_matrix = tfidf.fit_transform(df['Description'])

# print the shape of tfidf_matrix
print(tfidf_matrix.shape)

# create the cosine similarity between the vectors inside the tf-idf matrix.
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# drop the duplicates and create the indices based on the titles.
d = df.drop_duplicates(subset=['Title'])
indices = pd.Series(d.index, index=d['Title'])


# takes in a title, the cosine_sim and the indices.
# returns the top 5 recommendations above 0.3
def content_recommender(title, cosine_sim=cosine_sim, indices=indices):
    # get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwise similarity scores of all movies with that movie
    # And convert it into a list of tuples as described above
    sim_scores = list(enumerate(cosine_sim[idx]))

    # if the similarity is above 0.3 add the item to the 'sort' list
    sort = []
    for item in sim_scores:
        if item[1] > 0.3:
            sort.append(item)
    # Sort the list
    sort = sorted(sort, key=lambda x: x[1], reverse=True)
    # Get the scores of the 5 most similar courses. Ignore the first course, as this is the same as the title.
    sim_scores = sort[1:6]

    indxs = []
    for item in sim_scores:
        indxs.append(item[0])

    # Return the top 5 most similar courses
    return d['Title'].iloc[indxs]


# test the recommendation function
print(content_recommender('Emotional Intelligence'))

# Because we're using a large dataset K-Means is too slow, so we're using MiniBatchKMeans
# Here we find the 'k' variable by looking for an 'elbow'
kmeans_per_k = [MiniBatchKMeans(n_clusters=k, random_state=42).fit(tfidf_matrix)
                for k in range(1, 25)]
WCSS = [model.inertia_ for model in kmeans_per_k]

# Plot the inertias to find the elbow
plt.figure()
plt.plot(range(1, 25), WCSS)
plt.xlabel("$k$", fontsize=14)
plt.ylabel("WCSS", fontsize=14)
plt.annotate('Elbow',
             xy=(11, WCSS[3]),
             xytext=(0.55, 0.55),
             textcoords='figure fraction',
             fontsize=16,
             arrowprops=dict(facecolor='black', shrink=0.1)
             )
plt.axis([1, 25, 0, 6000])
plt.show()

# We use TruncatedSVD because our data (tfidf_matrix) is sparse. We've selected 25 components because that seemed to be
# give the best results between.
# TSNE is then used to create 2D data that can be plotted.

smaller_df = TSNE().fit_transform(TruncatedSVD(n_components=25, random_state=42).fit_transform(tfidf_matrix))
print(smaller_df)

# Based on the 'elbow' we found that there are 11 clusters.
k = 11
kmeans = MiniBatchKMeans(n_clusters=k, random_state=42)
y_pred = kmeans.fit_predict(tfidf_matrix)

# Save the data to use in the object_locator, which is a streamlit app.
SAVE = True
if SAVE:
    pd.DataFrame(smaller_df).to_csv("./model/coordinates.csv")
    pd.DataFrame(y_pred).to_csv("./model/prediction.csv")
    indices.to_csv("./model/indices.csv")


def plot_clusters(X, y=None):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='viridis')
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()


# Plot the clusters
plot_clusters(smaller_df, y=y_pred)


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
        plt.axis([target[0] - zoom_axis, target[0] + zoom_axis, target[1] - zoom_axis, target[1] + zoom_axis])
    plt.show()


# Plot the clusters with a 'target'. This is a title of a course
target_point = get_coordinates(indices, smaller_df, 'Eager to help? ~ Share your expertise (WIP)')
plot_clusters_with_target(smaller_df, target_point, y_pred)
plot_clusters_with_target(smaller_df, target_point, y_pred, zoom_axis=30)
