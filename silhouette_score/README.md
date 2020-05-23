This is my implementation for fast computing silhouette score.

Silhouette_score_parallel.py implements silhouette score using parallel computing based on a blocking strategy, and can work with any customized metrics that sklearn package doesn't complie. Besides, it returns average intra distances and inter distances over all samples to evaluate a cluster result.

Pairwise_distance.py calculate affince gap distance between two given samples. Any customized metrics/distance method should work with the silhouette function above.

Reference: https://gist.github.com/AlexandreAbraham/5544803
