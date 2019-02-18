import os
from keras.models import load_model
import keras.losses
import keras.backend as K
from scipy import sparse
from sklearn.externals import joblib
import numpy as np
import normalize_hindex_prob as prob


class StandaloneHindexNormalizer():

    def __init__(self, prob, model_file, paperx_file, scaler_file):

        self.prob = prob

        # Load keras model
        # This is *not* the actual loss function, but that is not needed for making predictions
        keras.losses.myloss = keras.losses.mean_squared_error
        self.model = load_model(model_file)
        self.input_shape = self.model.get_layer(index=0).input_shape

        # Load input data
        self.paperx = sparse.load_npz(paperx_file)

        # Load scaler
        scaler = joblib.load(scaler_file)

        # Scale inputs
        self.paperx = scaler.transform(self.paperx)

        # Prepare normalization function
        h0 = K.placeholder(ndim=2)
        param0 = K.placeholder(ndim=2)
        param1 = K.placeholder(ndim=2)
        self.normalize = K.function(
            [h0, param0, param1],
            [self.prob.normalized_hindex(h0, param0, param1)])

    def predict_and_normalize_by_papers(self, by_papers, h0):

        nauthors = len(by_papers)
        nfeatures = self.input_shape[2]
        npapers = self.input_shape[1]

        # Generate input data
        x = np.zeros((nauthors, npapers, nfeatures))
        for i, papers in enumerate(by_papers):
            x[i, :len(papers), :] = self.paperx[papers].toarray()
        xaux = np.zeros((nauthors, 0))

        # Predict probability distribution parameters
        y_net = self.model.predict({'perpaper_inputs': x,
                                    'perauthor_inputs': xaux})

        # Calculate normalized h-index
        normalized_hindex = self.normalize(
            [h0[:, 0], y_net[:, 0], y_net[:, 1]])[0]

        return y_net, normalized_hindex


if __name__ == '__main__':

    # Paths to various files
    data_dir = os.path.join('data', 'normalized-hindex')
    model_file = os.path.join(data_dir, 'models', 'net-gamma.h5')
    paperx_file = os.path.join(data_dir, 'net_data_paperx.npz')
    scaler_file = os.path.join(data_dir, 'scaler')

    # Chooses gamma probability distribution function
    prob = prob.ProbGamma()

    # Preloads model and per-paper data
    normalizer = StandaloneHindexNormalizer(prob, model_file, paperx_file,
                                            scaler_file)

    from timeit import default_timer as t
    for i in range(0, 10000):
        start = t()

        # Two dummy authors specified by dummy data
        # by_papers: paper ids for each author, h0: h0 for each author
        by_papers = np.array([ [0, 1, 2, 3, 4], [50, 51, 52, 53, 150, 10054] ])
        h0 = np.array([[1], [3]])

        # Get probability distribution parameters and normalized h-index
        y_net, normalized_hindex = normalizer.predict_and_normalize_by_papers(
            by_papers, h0)
        print('Probability distribution parameters', y_net)
        print('Normalized h-index', normalized_hindex)
        print('log(Normalized h-index)', np.log(normalized_hindex))

        end = t()
        print(i, "Time: ", end-start)

