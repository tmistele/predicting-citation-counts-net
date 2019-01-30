from net import Net
from datetime import date
import numpy as np
import os
import keras.backend as K
import psycopg2
import settings
from inspect import signature
import pandas as pd
from scipy import sparse


def prob_gammapoisson(alpha, beta, h0):
    def gamma(a):
        tf = K.tensorflow_backend.tf
        return tf.exp(tf.lgamma(a))

    # http://www.math.wm.edu/~leemis/chart/UDR/PDFs/Gammapoisson.pdf
    return (gamma(h0 + beta) * alpha**h0) / \
        (gamma(beta) * (1+alpha)**(beta+h0) * gamma(1+h0))


def prob_lognormal(mu, sigma, h0):
    # NOTE: We add .001 to h0, since keras/tf cannot handle our prob_lognormal
    # for h0 -> 0 very well. If we don't do this, we tend to always get
    # loss=nan
    h0 += .001

    erfc = K.tensorflow_backend.tf.math.erfc
    def lognormal_cum(x):
        # https://en.wikipedia.org/wiki/Log-normal_distribution#Cumulative_distribution_function
        return (1/2) * erfc(-(K.log(x) - mu)/(np.sqrt(2) * sigma))

    # Log-normal probability mass function
    return lognormal_cum(h0+1) - lognormal_cum(h0)


def prob_gamma(alpha, beta, h0):
    # NOTE: We add .001 to h0, since keras/tf cannot handle our prob_gamma for
    # h0 -> 0 very well. If we don't do this, we tend to always get loss=nan
    h0 += .001

    # https://www.tensorflow.org/api_docs/python/tf/math/igamma
    # igamma(a, x) = 1/Gamma(a) * int_0^x (t^(a-1) exp(-t))
    igamma = K.tensorflow_backend.tf.math.igamma
    def gamma_cum(x):
        # https://en.wikipedia.org/wiki/Gamma_distribution
        return igamma(alpha, beta*x)

    # Gamma probability mass function
    return gamma_cum(h0+1) - gamma_cum(h0)


class NormalizedHindexNet(Net):

    # We don't want to predict the future, just analyze the status quo
    cutoff_date = date(2019, 1, 1)
    predict_after_years = 0

    data_positions = {
        'padding': 0,
        'num_authors': 1,
        'months': 2,
        'hotness_1': 3,
        'hotness_5': 4,
        'hotness_inf': 5,
        'categories': 6,
        }
    data_positions_aux = {
        }
    paper_topics_dim = 0
    numcategories = 38

    def __init__(self, prob):
        self._con = None
        self.data_dir = os.path.join(settings.DATA_DIR, 'normalized-hindex')
        self.prob = prob

        # Make sure directories exist
        os.makedirs(self.data_dir, exist_ok=True)

        # Directory for net x and y data may be set to a different location
        # (e.g. for running on cluster)
        self.data_dir_xy = self.data_dir

        # Used by cross-validation
        self.suffix_author_ids = ""

        self.target = 'hindex_cumulative'

        self.set_exclude_data([])

    def con(self):
        if self._con is None:
            self._con = psycopg2.connect(settings.PSQL_CONNECT_STRING)
        return self._con

    def save_train_authors(self):
        c = self.con().cursor()
        c.execute("SELECT COUNT(*) FROM authors")
        numauthors = c.fetchone()[0]

        data = np.zeros(numauthors,
                        dtype=[('author_id', 'i4'), ('train', 'i4')])

        data['author_id'][:] = np.arange(numauthors)

        # Take ~70% training
        numtraining = int(np.ceil(numauthors * .7 / 500.) * 500.)
        train_indices = np.random.permutation(numauthors)[0:numtraining-1]
        data['train'][train_indices] = 1

        np.save(self.get_author_ids_filename(), data)

    def get_net_filename(self):
        return os.path.join(self.data_dir, 'net.h5')

    def get_data_author_generator(self):
        authors = self.get_train_authors()
        for i, author in enumerate(authors):
            yield i, author['author_id']

    def generate_data_y(self):
        print("Generating y data...")

        author_ids = [author_id
                      for _, author_id in self.get_data_author_generator()]

        # Output data
        y = np.zeros(len(author_ids))

        c = self.con().cursor()

        authors = ','.join([str(id) for id in author_ids])

        print("Fetching h-indices from elementary_metrics...")
        c.execute("""
            SELECT value, author_id
            FROM elementary_metrics
            WHERE index = 0
            ORDER BY author_id ASC""")
        tmp = {}
        for (value, author_id) in c:
            tmp[author_id] = value

        print("Filling y array...")
        for i, author_id in enumerate(author_ids):
            hindex = tmp[author_id]
            print(i, author_id, hindex)
            y[i] = hindex

        columns = ['y' + str(i) for i in range(0, self.predict_after_years+1)]
        return pd.DataFrame(y, columns=columns)

    def generate_data_x(self):

        print("Generating x data...")

        import cbor2, yaml

        print("Loading author papers...")
        author_papers = cbor2.load(open(
            os.path.join(settings.DATA_DIR, 'arxiv', 'keywords-backend',
                         'nn_data', 'author_papers.cbor'),
            'rb'
        ))
        numauthors = len(author_papers)
        effective_max_papers = len(max(author_papers, key=lambda x: len(x)))
        if effective_max_papers != 1238:
            # TODO
            raise Exception("Note to self: Modify also self.load_data_x")
        print("#authors", numauthors)
        print("Effective max papers", effective_max_papers)

        author_ids = [author_id
                      for _, author_id in self.get_data_author_generator()]

        xcolumns = self.get_column_names_x(exclude=False)
        xauxcolumns = self.get_column_names_xaux(exclude=False)

        # Each paper (at the moment) has only 1 category
        # So there will only be exactly 1 non-zero value in our one-hot vector
        # Therefore, we can save a lot of memory by saving only that one value
        # in our sparse matrix
        if self.numcategories:
            sparse_xcolumns = len(xcolumns) - self.numcategories + 1
        else:
            sparse_xcolumns = len(xcolumns)

        # Per-paper input data
        # Prepare for sparse array with shape (nauthors*npapers, nfeatures)
        print("Preparing sparse rows/cols/data...")
        nactualpapers = 0
        for author_id in author_ids:
            nactualpapers += len(author_papers[author_id])
        rows = np.concatenate([i*np.ones(sparse_xcolumns)
                               for i in range(0, nactualpapers)])
        # NOTE: The column is not correct for the 'category', which is
        # corrected in the loop below
        cols = np.tile(np.arange(sparse_xcolumns), nactualpapers)
        data = np.zeros(rows.shape)

        # For #coauthors
        print("Loading paper authors...")
        paper_authors = cbor2.load(open(
            os.path.join(settings.DATA_DIR, 'arxiv', 'keywords-backend',
                         'nn_data', 'paper_authors.cbor'),
            'rb'
        ))

        # For months
        print("Loading paper dates...")
        paper_dates = cbor2.load(open(
            os.path.join(settings.DATA_DIR, 'arxiv', 'keywords-backend',
                         'nn_data', 'paper_dates.cbor'),
            'rb'
        ))

        # For hotness
        paper_hotness1 = cbor2.load(open(
            os.path.join(settings.DATA_DIR, 'arxiv', 'keywords-backend',
                         'nn_data', 'paper_keyword_hotness_1year.cbor'),
            'rb'
        ))
        paper_hotness5 = cbor2.load(open(
            os.path.join(settings.DATA_DIR, 'arxiv', 'keywords-backend',
                         'nn_data', 'paper_keyword_hotness_5year.cbor'),
            'rb'
        ))
        paper_hotnessinf = cbor2.load(open(
            os.path.join(settings.DATA_DIR, 'arxiv', 'keywords-backend',
                         'nn_data', 'paper_keyword_hotness_alltime.cbor'),
            'rb'
        ))


        # For categories
        print("Loading (paper) categories...")
        # astro.CO -> astro etc.
        categories = yaml.load(open(
            os.path.join(settings.DATA_DIR, 'arxiv', 'keywords-backend',
                         'nn_data', 'categories.yaml'),
            'rb'
        ))
        tmp = list(set([category.split('.')[0] for category in categories]))
        tmp = {value: i for i, value in enumerate(tmp)}
        if(len(tmp) != self.numcategories):
            raise Exception("len(categories) != self.numcategories")
        categories = {id: tmp[name.split('.')[0]]
                      for id, name in enumerate(categories)}

        paper_categories = cbor2.load(open(
            os.path.join(settings.DATA_DIR, 'arxiv', 'keywords-backend',
                         'nn_data', 'paper_categories.cbor'),
            'rb'
        ))

        # Per-author input data
        xaux = np.zeros((numauthors, len(xauxcolumns)))

        datapos = 0
        for i, author_id in enumerate(author_ids):
            papers = author_papers[author_id]

            print(author_id, "has #papers:", len(papers))

            tmp = np.zeros((len(papers), sparse_xcolumns))

            # Explicitly tell net which data are padding and which are papers
            index = self.data_positions['padding']
            tmp[:, index] = np.ones(len(papers))

            # num_authors
            index = self.data_positions['num_authors']
            tmp[:, index] = np.array([len(paper_authors[paper])
                                      for paper in papers])

            # months
            index = self.data_positions['months']
            tmp[:, index] = np.array(
                [(self.cutoff_date -
                  date.fromtimestamp(paper_dates[paper])).days/30
                 for paper in papers])

            # Hotness 1, 5, inf
            index = self.data_positions['hotness_1']
            tmp[:, index] = np.array([paper_hotness1[paper]
                                      for paper in papers])
            index = self.data_positions['hotness_5']
            tmp[:, index] = np.array([paper_hotness5[paper]
                                      for paper in papers])
            index = self.data_positions['hotness_inf']
            tmp[:, index] = np.array([paper_hotnessinf[paper]
                                      for paper in papers])

            # categories
            # NOTE: Categories are saved as a one-hot vector in the sparse
            # matrix. However, we only save the '1' and not the '0's of this
            # one-hot vector to save memory. Therefore, we must adjust the
            # column to save this '1' to
            index = self.data_positions['categories']
            for paperi, paper in enumerate(papers):
                # The value to save is '1'
                # After flattening, this '1' ends up at
                # tmp.flatten()[paperi*tmp.shape[1] + index]
                tmp[paperi, index] = 1
                # The column to save this to is the category_index'th category
                # column (which start at 'index')
                category_index = categories[paper_categories[paper]]
                cols[datapos + paperi*tmp.shape[1] + index] =\
                    index + category_index

            # Translate to sparse matrix format
            tmp = tmp.flatten()
            data[datapos:datapos+len(tmp)] = tmp
            datapos += len(tmp)

        # Consistency check
        if datapos != len(data):
            raise Exception("datapos != len(data)")

        # Free memory
        print("Freeing memory...")
        del author_papers
        del paper_authors
        del paper_dates
        del paper_hotness1
        del paper_hotness5
        del paper_hotnessinf
        del paper_categories
        del tmp
        del author_ids

        import gc
        gc.collect()

        # Create sparse matrix for x
        x = sparse.csc_matrix(
            (data, (rows, cols)),
            shape=(numauthors*effective_max_papers, len(xcolumns)))

        # xaux to DataFrame
        xaux = pd.DataFrame(xaux, columns=xauxcolumns)

        # x to SparseDataFrame?
        # x = pd.SparseDataFrame(x)

        return x, xaux

    def load_data_x(self, indices=None, total_authors=None):
        file_x = os.path.join(self.data_dir_xy, 'net_data_x.npz')
        file_xaux = os.path.join(self.data_dir_xy, 'net_data_xaux.h5')

        if indices is not None:
            indices = indices[0]

        # Columns to load
        xcolumns = self.get_column_names_x(exclude=True)
        xauxcolumns = self.get_column_names_xaux(exclude=True)

        try:
            x = sparse.load_npz(file_x)

            try:
                xaux = pd.read_hdf(file_xaux, where=pd.IndexSlice[indices],
                                   columns=xauxcolumns)
            except ValueError as e:
                if len(xauxcolumns) == 0:
                    xaux = pd.DataFrame(np.zeros((x.shape[0], 0)), columns=[])
                else:
                    raise e

        except (FileNotFoundError, KeyError) as e:
            x, xaux = self.generate_data_x()
            xaux.to_hdf(file_xaux, key='xaux', format='table')
            sparse.save_npz(file_x, x)
            if indices is None:
                xaux = xaux[xauxcolumns]
            else:
                xaux = xaux.iloc[indices][xauxcolumns]

        # TODO: Support x columns?

        if indices is not None:
            nfeatures = x.shape[1]
            # TODO: Don't hardcode 1238
            npapers = 1238
            print("Restricting to selected authors...")
            x = x.reshape((-1, npapers*nfeatures)).tocsr()
            x = x[indices]
            x = x.reshape((-1, nfeatures)).tocsr()
        else:
            x = x.tocsr()

        return x, xaux

    def _scale_inputs(self, x, xaux, is_train_inputs):
        from sklearn.preprocessing import StandardScaler
        from sklearn.externals import joblib

        filename = os.path.join(self.data_dir, 'scaler')
        try:
            scaler = joblib.load(filename)
        except FileNotFoundError:
            if is_train_inputs:
                scaler = StandardScaler(copy=False, with_mean=False)
                scaler.partial_fit(x)
                joblib.dump(scaler, filename)
            else:
                raise Exception("Scaler not found")

        x = scaler.transform(x)

        scale_fields_aux = [
            ]
        for field in scale_fields_aux:
            column = field + "0"
            if column not in xaux.columns and field in self.__exclude_data:
                continue
            filename = os.path.join(self.data_dir,
                                    'scaler-%s' % field)
            try:
                scaler = joblib.load(filename)
            except FileNotFoundError:
                if is_train_inputs:
                    scaler = StandardScaler(copy=False)
                    scaler.fit(xaux[[column]])
                    joblib.dump(scaler, filename)
                else:
                    raise Exception("Scaler not found")

            xaux[[column]] = scaler.transform(xaux[[column]])

    def train(self, activation='tanh', load=False):

        # Speed up batch evaluations
        self.clear_keras_session()

        # Load data
        print("Loading net training data...")
        x, xaux, y = self.load_train_data()

        # How many parameters does our probability distribution have?
        num_prob_params = len(signature(self.prob).parameters)-1

        # Zero-padding for y
        # ytmp = np.zeros((y.shape[0], num_prob_params))
        # ytmp[:, 0] = y.flatten()
        # y = ytmp

        # Prepare generator
        from math import ceil
        nauthors = y.shape[0]
        author_batch_size = 50
        batches_per_epoch = ceil(nauthors / author_batch_size)
        nfeatures = x.shape[1]
        npapers = x.shape[0]//nauthors
        if x.shape[0] % nauthors:
            raise Exception("Non-integer npapers")
        def generator():
            while True:
                start = 0
                end = author_batch_size
                for i in range(0, batches_per_epoch):

                    # Make dense matrix from csr matrix
                    xbatch = x[start*npapers:end*npapers].toarray()

                    # Reshape to (nauthors, npapers, nfeatures)
                    xbatch = xbatch.reshape((-1, npapers, nfeatures))

                    yield ({'perpaper_inputs': xbatch,
                            'perauthor_inputs': xaux[start:end]},
                            y[start:end])
                    start += author_batch_size
                    end += author_batch_size


        # Custom loss function
        def loss(y_true, y_pred, debug=False):
            # y_true contains the h-index at index 0 and is then (possibly)
            # padded with zeros
            h0 = y_true[:, 0]
            if debug:
                h0 = K.print_tensor(h0, message='h0 = ')

            # y_pred contains the parameters of the probability distribution
            if debug:
                y_pred = K.print_tensor(y_pred, message='y_pred = ')

            # Both need to be positive
            if num_prob_params != 2:
                raise Exception("Not implemented")
            alpha = K.exp(y_pred[:, 0])
            beta = K.exp(y_pred[:, 1])
            if debug:
                alpha = K.print_tensor(alpha, 'alpha = ')
                beta = K.print_tensor(beta, 'beta = ')

            # Probability
            p = self.prob(alpha, beta, h0)
            if debug:
                p = K.print_tensor(p, message='p = ')
            # Sometimes we get p = 0 due to limited numerical precision, which
            # is bad for the log in the next step
            p = K.clip(p, K.epsilon(), 1.)
            if debug:
                p = K.print_tensor(p, message='p(post-clip) = ')

            # -log(p_{alpha,beta}(h0))
            logp = -K.log(p)
            if debug:
                logp = K.print_tensor(logp, message='logp = ')

            mean = K.mean(logp, axis=-1)
            if debug:
                mean = K.print_tensor(mean, message='mean = ')

            return mean

        # print(K.eval(loss(K.variable(value=np.array([[15, 0], [10, 0]])), K.variable(value=np.array([[1.5, 2.3], [2.6, 2.7]])))))
        # Should give 4.66331 for gammapoisson

        # Build/load keras model
        if load:
            print("Loading keras / load model...")
            model = self.load_model()
        else:
            print("Loading keras / build model...")
            from keras.models import Model
            from keras.layers import Input, Dense, Conv1D, concatenate, \
                GlobalAveragePooling1D

            perpaper_inputs = Input(shape=(npapers, nfeatures),
                                    name='perpaper_inputs')
            perauthor_inputs = Input(shape=xaux[0].shape,
                                     name='perauthor_inputs')

            tmp = Conv1D(
                filters=70,
                kernel_size=1,
                strides=1,
                activation=activation,
                input_shape=x[0].shape)(perpaper_inputs)

            tmp = GlobalAveragePooling1D()(tmp)

            tmp = concatenate([tmp, perauthor_inputs])
            tmp = Dense(units=70, activation=activation)(tmp)

            outputs = Dense(units=num_prob_params,
                            activation='relu')(tmp)

            model = Model(inputs=[perpaper_inputs, perauthor_inputs],
                          outputs=outputs)

            model.compile(
                loss=loss, # Use custom loss function
                optimizer='adam')

        # Fit keras model
        model.fit_generator(generator(), epochs=self.epochs,
                            steps_per_epoch=batches_per_epoch)
        model.save(self.get_net_filename())


if __name__ == '__main__':


    # print(K.eval(prob_gamma(K.variable(value=5.0), K.variable(value=1.5), K.variable(value=3))))
    # Should give 0.247047

    # print(K.eval(prob_lognormal(K.variable(value=5.0), K.variable(value=1.5), K.variable(value=3))))
    # Should give 0.0033465

    # n = NormalizedHindexNet(prob=prob_gammapoisson)
    # n = NormalizedHindexNet(prob=prob_lognormal)
    n = NormalizedHindexNet(prob=prob_gamma)
    n.train()
