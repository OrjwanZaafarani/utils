import sklearn
from sklearn.cluster import KMeans

def cluster(df_train, df_val, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters).fit(df_train[['bbox_width','bbox_height']])
    y_hat = kmeans.predict(df_train[['bbox_width','bbox_height']])
    df_train['w_h_cluster'] = y_hat
    centroids_train = kmeans.cluster_centers_
    min_train_batch_size = df_train.groupby(['w_h_cluster']).count()['index'].mean()
    
    kmeans = KMeans(n_clusters=n_clusters).fit(df_val[['bbox_width','bbox_height']])
    y_hat = kmeans.predict(df_val[['bbox_width','bbox_height']])
    df_val['w_h_cluster'] = y_hat
    centroids_val = kmeans.cluster_centers_
    min_val_batch_size = df_val.groupby(['w_h_cluster']).count()['index'].mean()

    return df_train, df_val, centroids_train, centroids_val, min_train_batch_size, min_val_batch_size
