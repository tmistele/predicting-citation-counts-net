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
from abc import ABC, abstractmethod


class Prob(ABC):
    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def prob(self, h0, *args):
        pass

    @abstractmethod
    def cum(self, h0, *args):
        pass

    def normalized_hindex(self, h0, *args):
        # 1/P(h>h0) = 1/(1-P(h<=h0))
        return 1/(1-self.cum(h0, *args))

    @property
    def num_params(self):
        # How many parameters does our probability distribution have?
        return len(signature(self.prob).parameters)-1


class ProbGammaPoisson(Prob):
    @property
    def name(self):
        return 'gamma-poisson'

    def gamma(self, a):
        tf = K.tensorflow_backend.tf
        return tf.exp(tf.lgamma(a))

    def cum(self, h0, alpha, beta):
        raise Exception("Not implemented yet")

    def prob(self, h0, alpha, beta):
        # http://www.math.wm.edu/~leemis/chart/UDR/PDFs/Gammapoisson.pdf
        return (self.gamma(h0 + beta) * alpha**h0) / \
            (self.gamma(beta) * (1+alpha)**(beta+h0) * self.gamma(1+h0))


class ProbLognormal(Prob):
    @property
    def name(self):
        return 'lognormal'

    def _lognormal_cum(self, x, mu, sigma):
        erfc = K.tensorflow_backend.tf.math.erfc
        # https://en.wikipedia.org/wiki/Log-normal_distribution#Cumulative_distribution_function
        return (1/2) * erfc(-(K.log(x) - mu)/(np.sqrt(2) * sigma))

    def cum(self, h0, mu, sigma):
        return self._lognormal_cum(h0+1, mu, sigma) -\
            self._lognormal_cum(0., mu, sigma)

    def prob(self, h0, mu, sigma):
        # Log-normal probability mass function
        # NOTE: We add .001 to h0, since keras/tf cannot handle our
        # prob_lognormal for h0 -> 0 very well. If we don't do this, we tend to
        # always get loss=nan
        return self._lognormal_cum(h0+1+.001, mu, sigma) -\
            self._lognormal_cum(h0+.001, mu, sigma)


class ProbGamma(Prob):
    @property
    def name(self):
        return 'gamma'

    def _gamma_cum(self, x, alpha, beta):
        # https://www.tensorflow.org/api_docs/python/tf/math/igamma
        # igamma(a, x) = 1/Gamma(a) * int_0^x (t^(a-1) exp(-t))
        igamma = K.tensorflow_backend.tf.math.igamma
        # https://en.wikipedia.org/wiki/Gamma_distribution
        return igamma(alpha, beta*x)

    def cum(self, h0, alpha, beta):
        return self._gamma_cum(h0+1, alpha, beta) -\
            self._gamma_cum( 0., alpha, beta)

    def prob(self, h0, alpha, beta):
        # Gamma probability mass function
        # NOTE: We add .001 to h0, since keras/tf cannot handle our prob_gamma
        # for h0 -> 0 very well. If we don't do this, we tend to always get
        # loss=nan Gamma probability mass function
        return self._gamma_cum(h0+1+.001, alpha, beta) -\
            self._gamma_cum(h0+.001, alpha, beta)


class NormalizedHindexNet(Net):

    epochs = 20

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

        print("Using prob", self.prob.name)

        # Make sure directories exist
        os.makedirs(os.path.join(self.data_dir, 'models'), exist_ok=True)

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
        return os.path.join(self.data_dir, 'models',
                            'net-%s.h5' % self.prob.name)

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
                    xaux = pd.DataFrame(np.zeros((len(indices), 0)))
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

    def get_loss_function(self, debug=False):
        def loss(y_true, y_pred):
            # y_true contains the h-index at index 0 and is then (possibly)
            # padded with zeros
            h0 = y_true[:, 0]
            if debug:
                h0 = K.print_tensor(h0, message='h0 = ')

            # y_pred contains the parameters of the probability distribution
            if debug:
                y_pred = K.print_tensor(y_pred, message='y_pred = ')

            # Both need to be positive
            if self.prob.num_params != 2:
                raise Exception("Not implemented num_prob_params = "+
                                self.prob.num_params)
            alpha = y_pred[:, 0]
            beta = y_pred[:, 1]
            if debug:
                alpha = K.print_tensor(alpha, 'alpha = ')
                beta = K.print_tensor(beta, 'beta = ')

            # Probability
            p = self.prob.prob(h0, alpha, beta)
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

        return loss

    def init_loss_function(self):
        # Make loading models with custom loss function work
        # https://github.com/keras-team/keras/issues/5916#issuecomment-290344248
        import keras.losses
        keras.losses.myloss = self.get_loss_function()

    def load_model(self):
        self.init_loss_function()
        return super().load_model()

    def get_generator(self, x, xaux, y):
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

        return generator, batches_per_epoch, nauthors, npapers, nfeatures

    def train(self, activation='tanh', load=False):

        # Speed up batch evaluations
        self.clear_keras_session()

        # Load data
        print("Loading net training data...")
        x, xaux, y = self.load_train_data()

        # Zero-padding for y
        # ytmp = np.zeros((y.shape[0], self.prob.num_params))
        # ytmp[:, 0] = y.flatten()
        # y = ytmp

        # Prepare generator
        generator, batches_per_epoch, nauthors, npapers, nfeatures = \
            self.get_generator(x, xaux, y)

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

            outputs = Dense(units=self.prob.num_params,
                            activation='softplus')(tmp)

            model = Model(inputs=[perpaper_inputs, perauthor_inputs],
                          outputs=outputs)

            self.init_loss_function()

            model.compile(
                loss='myloss', # Use custom loss function
                optimizer='adam')

        # Fit keras model
        model.fit_generator(generator(), epochs=self.epochs,
                            steps_per_epoch=batches_per_epoch)
        model.save(self.get_net_filename())

    def do_evaluate(self, y, y_net, normalized_hindex):

        # Loss
        lossfunc = self.get_loss_function()
        lossval = K.eval(lossfunc(y, y_net))
        print(self.prob.name)
        print("-> Loss", lossval)

        # Prepare normalized hindex such that it is roughly straight line in a
        # h0-normalizedhindex plot
        if self.prob.name == 'lognormal':
            normalized_hindex = np.power(normalized_hindex, .33)
            name = 'normalized hindex^(.33)'
        elif self.prob.name == 'gamma':
            normalized_hindex = np.log(normalized_hindex)
            name = 'log(normalized hindex)'

        print("Plotting...", y.shape[0])
        import matplotlib
        matplotlib.use('PDF')
        import matplotlib.pyplot as plt

        # Scatter plot parameters
        print(y_net[0:10, 0])
        plt.hist(y_net[:, 0], bins=500)[0]
        plt.ylabel('#')
        plt.xlabel('param0')
        filename = self.get_net_filename().replace('.h5',
                                                   '-validation-param0.png')
        plt.savefig(filename)
        plt.close()

        print(y_net[0:10, 1])
        plt.hist(y_net[:, 1], bins=500)[0]
        plt.ylabel('#')
        plt.xlabel('param1')
        filename = self.get_net_filename().replace('.h5',
                                                   '-validation-param1.png')
        plt.savefig(filename)
        plt.close()

        # Scatter plot h0 vs normalized hindex
        plt.scatter(y[:, 0], normalized_hindex, s=.5, alpha=.2)
        plt.xlabel('h0')
        plt.ylabel(name)
        filename = self.get_net_filename().replace('.h5',
                                                   '-validation-results.png')
        plt.savefig(filename)

        plt.xlim(0, 4.5)
        plt.ylim(0.5, 3)
        plt.savefig(filename.replace('.png', '-zoomed.png'))

        plt.close()

        # Histograms for individual h0s
        import scipy.stats as st
        binsx = np.arange(np.max(y[:, 0]))
        max = 50
        binsy = np.linspace(0, max, np.ceil(max/0.01))
        hist, *_ = st.binned_statistic_2d(y[:, 0], normalized_hindex, None,
                                          'count',
                                          bins=(binsx, binsy))

        for h0 in [0, 3, 5, 10]:
            indices = np.where(hist[h0]>0)
            plt.title('h0 = '+str(h0))
            plt.ylabel('log10(#authors)')
            plt.xlabel(name)
            plt.plot(binsy[indices], np.log10(hist[h0][indices]))
            filename = self.get_net_filename().replace(
                '.h5', '-validation-results-h0-%s-.png' % h0)
            plt.savefig(filename)
            plt.close()

        return lossval

    def evaluate(self):

        filename = self.get_net_filename().replace('.h5',
                                                   '-validation-results.npz')
        try:
            tmp = np.load(filename)
            y = tmp['y_true']
            y_net = tmp['y_pred']
            normalized_hindex = tmp['normalized_hindex']
        except FileNotFoundError:
            # Load data
            print("Loading net evaluation data...")
            x, xaux, y = self.load_validation_data()

            # Speed up batch evaluations
            self.clear_keras_session()

            # Prepare generator
            generator, batches_per_epoch, *_ = \
                self.get_generator(x, xaux, y)

            model = self.load_model()

            print("Predicting...")
            y_net = model.predict_generator(generator(),
                                            steps=batches_per_epoch)

            if self.prob.num_params != 2:
                raise Exception("Not implemented yet")
            normalized_hindex = K.eval(self.prob.normalized_hindex(
                K.variable(value=y[:, 0]),
                K.variable(value=y_net[:, 0]),
                K.variable(value=y_net[:, 1])))

            np.savez(filename, y_true=y, y_pred=y_net,
                     normalized_hindex=normalized_hindex)

        return self.do_evaluate(y, y_net, normalized_hindex)


if __name__ == '__main__':


    # Test for prob
    # print(K.eval(ProbGamma().prob(K.variable(value=3-.001), K.variable(value=5.0), K.variable(value=1.5))))
    # Should give 0.247047
    # print(K.eval(ProbLognormal().prob(K.variable(value=3-.001), K.variable(value=5.0), K.variable(value=1.5))))
    # Should give 0.0033465

    # Test for cum
    # h0 = K.variable(value=3.)
    # a = K.variable(value=1.)
    # b = K.variable(value=1.)
    # prob = ProbLognormal() # Should give 0.9816844
    # prob = ProbGamma() # Should give 0.650361
    # p = prob.prob
    # c = prob.cum
    # print(K.eval(p(0., a,b)+p(1., a,b)+p(2., a, b)+p(3.,a,b)))
    # print(K.eval(c(h0, a, b)))

    # Test for normalize_hindex
    # prob = ProbLognormal() # Should give 2.86009
    # prob = ProbGamma() # Should give 54.598
    # print(K.eval(prob.normalized_hindex(K.variable(value=3.), K.variable(value=1.), K.variable(value=1.))))

    # n = NormalizedHindexNet(prob=ProbGammaPoisson())
    # n = NormalizedHindexNet(prob=ProbLognormal())
    # n = NormalizedHindexNet(prob=ProbGamma())
    # n.train()
    # n.train(load=True)
    # n.evaluate()

    import argparse
    from runner import *

    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['train', 'evaluate'])
    parser.add_argument('--prob', choices=['gamma', 'lognormal',
                                           'gammapoisson'])

    args = parser.parse_args()
    if args.prob == 'gamma':
        prob = ProbGamma()
    elif args.prob == 'lognormal':
        prob = ProbLognormal()
    elif args.prob == 'gammapoisson':
        prob = ProbGammaPoisson()
    else:
        raise Exception("Invalid prob")

    n = NormalizedHindexNet(prob=prob)
    if args.action == 'train':
        n.train()
    elif args.action == 'evaluate':
        n.evaluate()
    else:
        raise Exception("Invalid action")

