from sklearn import preprocessing
from sklearn.decomposition import PCA

def PCA_Whiten(features, dim=64, copy=True, whiten=False):
    features = preprocessing.normalize(features, copy=copy,norm='l2')
    pca = PCA(n_component=dim, whiten=whiten, copy=copy)
    features = pca.fit_transform(features)
    features = preprocessing.normalize(feastures, copy=copy, norm='l2')
    return features
