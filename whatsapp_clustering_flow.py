from metaflow import FlowSpec, step, Parameter, IncludeFile
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import re

class WhatsAppClusteringFlow(FlowSpec):

    # Parameter untuk file teks
    data_file = IncludeFile(
        "data_file",
        help="Text file containing WhatsApp data",
        is_text=True,
    )

    n_clusters = Parameter(
        "n_clusters",
        help="Number of clusters for KMeans",
        default=3,
    )

    @step
    def start(self):
        """
        Load and preprocess the WhatsApp data.
        """
        # Parsing the WhatsApp chat file into DataFrame
        messages = []
        for line in self.data_file.splitlines():
            match = re.match(r'\[(.*?)\] (.*?): (.*)', line)
            if match:
                timestamp, sender, message = match.groups()
                messages.append({"timestamp": timestamp, "sender": sender, "message": message})
        
        self.data = pd.DataFrame(messages)
        print(f"Data loaded with {len(self.data)} messages.")
        self.next(self.clean_data)

    @step
    def clean_data(self):
        """
        Clean the message data.
        """
        def clean_text(text):
            return re.sub(r"[^a-zA-Z0-9 .,!?]", "", text)

        # Apply cleaning
        self.data["cleaned_message"] = self.data["message"].apply(clean_text)
        print(f"Data cleaning completed. Sample: {self.data['cleaned_message'].head()}")
        self.next(self.cluster_data)

    @step
    def cluster_data(self):
        """
        Perform clustering using KMeans.
        """
        # Vectorize the cleaned text
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(self.data["cleaned_message"].dropna())

        # Apply KMeans
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.data["cluster"] = kmeans.fit_predict(X)

        # Extract top words per cluster
        self.top_words = self._get_top_words_per_cluster(vectorizer, kmeans)
        print(f"Clustering into {self.n_clusters} clusters completed.")
        self.next(self.end)

    def _get_top_words_per_cluster(self, vectorizer, kmeans):
        """
        Extract top words for each cluster.
        """
        import numpy as np
        top_words = {}
        terms = vectorizer.get_feature_names_out()

        for i in range(kmeans.n_clusters):
            idx = np.argsort(kmeans.cluster_centers_[i])[::-1][:3]
            top_words[i] = [terms[j] for j in idx]

        return top_words

    @step
    def end(self):
        """
        Display the clustering results.
        """
        print("Flow finished successfully!")
        print("Top words per cluster:")
        for cluster, words in self.top_words.items():
            print(f"Cluster {cluster}: {', '.join(words)}")

if __name__ == "__main__":
    WhatsAppClusteringFlow()