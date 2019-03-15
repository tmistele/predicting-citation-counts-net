import os
from keras.models import load_model
import keras.losses
import keras.backend as K
from scipy import sparse
from sklearn.externals import joblib
import numpy as np


class StandaloneScimeterNet():

    def __init__(self, model_file, paperx_file, paperx_future_file,
                 scaler_file):

        # Load keras model
        self.model = load_model(model_file)
        self.input_shape = self.model.get_layer(index=0).input_shape

        # Load input data
        self.paperx = sparse.load_npz(paperx_file)
        self.paperx_future = sparse.load_npz(paperx_future_file)

        # Load scaler
        scaler = joblib.load(scaler_file)

        # Scale inputs
        self.paperx = scaler.transform(self.paperx)
        self.paperx_future = scaler.transform(self.paperx_future)

    def predict_by_papers(self, by_papers, future):

        nauthors = len(by_papers)
        nfeatures = self.input_shape[2]
        npapers = self.input_shape[1]

        # Use different input data for predicting 2018 than 2028, include
        # citations only until cutoff etc.
        paperx = self.paperx_future if future else self.paperx

        # Generate input data
        x = np.zeros((nauthors, npapers, nfeatures))
        for i, papers in enumerate(by_papers):
            x[i, :len(papers), :] = paperx[papers].toarray()
        xaux = np.zeros((nauthors, 0))

        # Predict
        y_net = self.model.predict({'perpaper_inputs': x,
                                    'perauthor_inputs': xaux})
        y_net = np.cumsum(y_net, axis=1)

        return y_net


if __name__ == '__main__':

    # Paths to various files
    data_dir = os.path.join('data', 'scimeter-net')
    model_file = os.path.join(data_dir, 'models', 'net.h5')
    paperx_file = os.path.join(data_dir, 'net_data_paperx.npz')
    paperx_future_file = os.path.join(data_dir, 'net_data_paperx-future.npz')
    scaler_file = os.path.join(data_dir, 'scaler')

    # Preloads model and per-paper data
    normalizer = StandaloneScimeterNet(model_file, paperx_file,
                                       paperx_future_file, scaler_file)

    from timeit import default_timer as t
    for i in range(0, 10000):
        start = t()

        # Predict 2018 from data up to 2008
        # Two dummy authors specified by dummy data
        # by_papers: paper ids for each author, h0: h0 for each author
        # NOTE: Only include papers until cutoff (2008 here)
        by_papers_until_2008 = np.array([ [0, 1, 2, 3, 4],
                                          [50, 51, 52, 53, 150, 10054] ])
        y_net_2018 = normalizer.predict_by_papers(by_papers_until_2008,
                                                  future=False)
        print('Based on data up to 2008-01-01, predict 2018-01-01', y_net_2018)

        # Predict 2028 from data up to 2018
        # Two dummy authors specified by dummy data
        # by_papers: paper ids for each author, h0: h0 for each author
        # NOTE: These are *more* papers than until 2008
        by_papers_until_2018 = np.array([ [0, 1, 2, 3, 4, 500, 501, 502],
                                          [50, 51, 52, 53, 150, 10054, 76, 78] ])
        y_net_2028 = normalizer.predict_by_papers(by_papers_until_2018,
                                                  future=True)
        print('Based on data up to 2018-01-01, predict 2028-01-01', y_net_2028)

        end = t()
        print(i, "Time: ", end-start)

