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

        # Save num citations before scaling them
        self.paper_ncit = self.paperx[:, 1]
        self.paper_ncit_future = self.paperx_future[:, 1]

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
        paper_ncit = self.paper_ncit_future if future else self.paper_ncit

        # Generate input data and calculate hindex at cutoff
        x = np.zeros((nauthors, npapers, nfeatures))
        hindex_at_cutoff = np.zeros(len(by_papers))
        for i, papers in enumerate(by_papers):
            x[i, :len(papers), :] = paperx[papers].toarray()

            ncits = paper_ncit[papers].toarray().flatten()
            ncits[::-1].sort()
            tmp = 0
            for ncit in ncits:
                if ncit <= tmp:
                    break
                tmp += 1
            hindex_at_cutoff[i] = tmp

        xaux = np.zeros((nauthors, 0))

        # Predict
        y_net = self.model.predict({'perpaper_inputs': x,
                                    'perauthor_inputs': xaux})
        # Add h-index at cutoff since the network predicts only differences
        # relative to this
        y_net[:, 0] += hindex_at_cutoff
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

        # Example for 401871. Can be used to cross-check
        by_papers = [
            [206071, 1353579, 1300487, 824919, 156782, 1305102, 188077, 319086,
             1304983, 1313818, 73831, 40732, 331550, 1306936, 145824, 319394,
             263587, 159841, 161574, 1356327, 354216, 605354, 582198, 489517,
             489519, 1294258, 1306803, 322740, 176822, 1294933, 1262631,
             1265052, 1350587, 1305198, 827966, 219808, 1295540, 390070,
             617975, 104398, 716496, 1311057, 144462, 182358, 1362775,
             647388, 1258461, 379111, 489507, 887523, 549103, 197862, 208102,
             457319, 457322, 1361819, 342637, 366446, 1309935, 1309937,
             1351462, 1359734, 1302263, 1307896, 168826, 1356923, 506979]
            ]
        print('Author 401871 (future)',
              normalizer.predict_by_papers(by_papers, future=True))

        end = t()
        print(i, "Time: ", end-start)

