from net import TimeSeriesNet
from datetime import date
from dateutil.relativedelta import relativedelta
from time import mktime
import numpy as np
import os
import keras.backend as K
import psycopg2
import settings
import pandas as pd
from scipy import sparse
from paperscape_importer import PaperscapeImporter
from db import db


class ScimeterNet(TimeSeriesNet):

    data_positions = {
        'padding': 0,
        'num_citations': 1,
        'months': 2,
        'length': 3,
        'published': 4,
        'jif': 5,
        'num_authors': 6,
        'categories': 7,
        }
    data_positions_aux = {
        }

    paper_topics_dim = 0
    numcategories = 38

    def __init__(self):
        self._con = None
        self.data_dir = os.path.join(settings.DATA_DIR, 'scimeter-net')

        # Indicate whether or not the 'future' or the training/validation data
        # should be loaded as input
        self._future = False

        # Make sure directories exist
        os.makedirs(os.path.join(self.data_dir, 'models'), exist_ok=True)

        # Directory for net x and y data may be set to a different location
        # (e.g. for running on cluster)
        self.data_dir_xy = self.data_dir

        # Used by cross-validation
        self.suffix_author_ids = ""

        self.target = 'hindex_cumulative'
        self.force_monotonic = True

        self.set_exclude_data([])

    def con(self):
        if self._con is None:
            self._con = psycopg2.connect(settings.PSQL_CONNECT_STRING)
        return self._con

    def get_cutoff_date(self):
        if self._future:
            return self.cutoff_date +\
                relativedelta(years=self.predict_after_years)
        else:
            return self.cutoff_date

    def _load_author_papers(self):
        print("Loading author papers...")
        import cbor2
        return cbor2.load(open(
            os.path.join(settings.DATA_DIR, 'arxiv', 'keywords-backend',
                         'nn_data', 'author_papers.cbor'),
            'rb'
        ))

    def _load_paper_dates(self):
        print("Loading paper dates...")
        import cbor2
        return cbor2.load(open(
            os.path.join(settings.DATA_DIR, 'arxiv', 'keywords-backend',
                         'nn_data', 'paper_dates.cbor'),
            'rb'
        ))

    def save_train_authors(self):
        c = self.con().cursor()
        c.execute("SELECT COUNT(*) FROM authors")
        numauthors = c.fetchone()[0]
        print("Total #authors", numauthors)

        data = np.zeros(numauthors,
                        dtype=[('author_id', 'i4'), ('train', 'i4')])

        data['author_id'][:] = np.arange(numauthors)

        # Take as training/validation authors only those with at least one
        # paper before the cutoff
        cutoff_date = mktime(self.get_cutoff_date().timetuple())
        print("The cutoff is", self.get_cutoff_date())

        print("Removing papers before cutoff...")
        author_papers = self._load_author_papers()
        paper_dates = self._load_paper_dates()
        author_papers = [
            [paper_id for paper_id in papers
             if paper_dates[paper_id] < cutoff_date]
            for papers in author_papers]

        print("Finding authors with at least one paper...")
        author_ids = np.array([
            author_id for author_id, papers in enumerate(author_papers)
            if len(papers) > 0])
        print("#authors with at least one paper", len(author_ids))

        # Initialize all authors as not being part of either validation or
        # training (train > 1)
        data['train'][:] = 2*np.ones(numauthors)

        # Set all authors with at least one paper as validation (train = 0)
        data['train'][author_ids] = np.zeros(len(author_ids))

        # Set ~70% as training
        numtraining = int(np.ceil(len(author_ids) * .7 / 500.) * 500.)
        train_indices = np.random.permutation(len(author_ids))[0:numtraining-1]
        data['train'][author_ids[train_indices]] = np.ones(len(train_indices))

        np.save(self.get_author_ids_filename(), data)

    def get_net_filename(self):
        return os.path.join(self.data_dir, 'models', 'net.h5')

    def get_data_author_generator(self):
        authors = self.get_train_authors()
        for i, author in enumerate(authors):
            yield i, author['author_id']

    def _import_ncit_from_paperscape(self):
        cl = db().cursor()
        try:
            cl.execute("""SELECT 1 FROM scimeter_net_citations LIMIT 1""")
            if len(cl.fetchall()) > 0:
                print("citations already imported")
                return
        except Exception:
            cl.execute("""
                CREATE TABLE `scimeter_net_citations` (
                    `cited_paper_id` int(11) NOT NULL,
                    `cited_date` date NOT NULL,
                    `citing_paper_id` int(11) NOT NULL,
                    `citing_date` date NOT NULL
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8;""")
            cl.execute("""
                ALTER TABLE `scimeter_net_citations`
                ADD KEY `cited_paper_citing_date`
                    (`cited_paper_id`,`citing_date`) USING BTREE;""")
            db().commit()

        print("Pre-loading papers...")
        c = self.con().cursor()
        c.execute("SELECT id, arxiv_id, timestamp FROM papers")
        paper_ids = {}
        for paper_id, arxiv_id, timestamp in c:
            paper_ids[arxiv_id] = (
                paper_id, date.fromtimestamp(timestamp).strftime("%Y-%m-%d"))

        print("Importing citations...")
        for citing, cited in PaperscapeImporter().generator():

            if citing not in paper_ids:
                continue
            citing_id, citing_date = paper_ids[citing]

            values = []
            for cited_arxiv_id in cited:
                if cited_arxiv_id not in paper_ids:
                    continue
                cited_id, cited_date = paper_ids[cited_arxiv_id]
                values.append(
                    '(%s, "%s", %s, "%s")' % (citing_id, citing_date,
                                              cited_id, cited_date)
                )

            if not len(values):
                continue

            print(citing, citing_id)

            cl.execute(
                "INSERT INTO scimeter_net_citations (" +
                "   citing_paper_id, citing_date, " +
                "   cited_paper_id, cited_date) " +
                "VALUES " + (', '.join(values)))

        db().commit()

        raise Exception("okay")

    def generate_data_y(self):

        self._import_ncit_from_paperscape()

        print("Generating y data...")

        author_ids = [author_id
                      for _, author_id in self.get_data_author_generator()]

        # Range of years
        if self._future:
            # For future we only need the h-index at cutoff
            years = range(0, 1)
        else:
            # We save the h-index at cutoff and 10 years in the future
            years = range(0, self.predict_after_years+1)

        # Output data
        y = np.zeros((len(author_ids), len(years)))

        author_papers = self._load_author_papers()

        cl = db().cursor()

        # SELECT part for citations at a given date
        # Note taht we do not need to restrict cited_date, since this is
        # automatically done since citing_date > cited_date.
        sel = []
        for year in years:
            cutoff = self.get_cutoff_date() + relativedelta(years=year)
            sel.append("SUM(IF(citing_date < '%s', 1, 0))" %
                cutoff.strftime("%Y-%m-%d"))
        sel = ',\n'.join(sel)

        print("Filling y array...")
        for i, author_id in enumerate(author_ids):
            print(i, author_id)

            if len(author_papers[author_id]) == 0:
                print("-> leave 0, no papers before cutoff")
                continue

            # Load ncit for papers for each year
            sql = """
                SELECT %s
                FROM scimeter_net_citations
                WHERE cited_paper_id IN (%s)
                GROUP BY cited_paper_id""" %\
                    (sel, ','.join([str(x) for x in author_papers[author_id]]))

            nresults = cl.execute(sql)
            result = np.fromiter(cl, count=nresults,
                dtype=[(str(i), 'i4') for i in years])

            # Calculate h-index for each year
            last_hindex = None
            for year in years:
                # Sort in descending order
                result[::-1][str(year)].sort()
                hindex = 0
                for ncit in result[str(year)]:
                    if ncit <= hindex:
                        break
                    hindex += 1

                if year == 0 or not self.force_monotonic:
                    y[i, year] = hindex
                else:
                    y[i, year] = hindex - last_hindex
                last_hindex = hindex

        columns = ['y' + str(i) for i in years]
        return pd.DataFrame(y, columns=columns)

    def load_data_y(self, indices=None):
        file_y = os.path.join(self.data_dir_xy,
                              'net_data_y%s.h5' %
                              ('-future' if self._future else ''))

        if indices is not None:
            indices = indices[0]

        try:
            y = pd.read_hdf(file_y, where=pd.IndexSlice[indices])
        except FileNotFoundError:
            y = self.generate_data_y()
            y.to_hdf(file_y, key='y', format='table')
            if indices is not None:
                y = y.iloc[indices]

        return y

    def _generate_paper_x(self):

        self._import_ncit_from_paperscape()

        print("Generating x data for all papers...")

        import cbor2, yaml

        xcolumns = self.get_column_names_x(exclude=False)

        # Each paper (at the moment) has only 1 category
        # So there will only be exactly 1 non-zero value in our one-hot vector
        # Therefore, we can save a lot of memory by saving only that one value
        # in our sparse matrix
        if self.numcategories:
            sparse_xcolumns = len(xcolumns) - self.numcategories + 1
        else:
            sparse_xcolumns = len(xcolumns)


        # For months
        paper_dates = self._load_paper_dates()

        ntotalpapers = len(paper_dates)

        # Load #citations as of self.cutoff_date or future
        print("Loading #citations...")
        paper_citations = np.zeros(ntotalpapers)
        cl = db().cursor()
        cl.execute("""
            SELECT cited_paper_id, COUNT(*)
            FROM scimeter_net_citations
            WHERE citing_date < %(cutoff_date)s
            GROUP BY cited_paper_id
            """, {'cutoff_date': self.get_cutoff_date()})
        for paper_id, num_citations in cl:
            paper_citations[paper_id] = num_citations

        # Load jifs, lengths
        print("Loading jifs, length...")
        paper_jifs = np.zeros(ntotalpapers)
        paper_lengths = np.zeros(ntotalpapers)
        c = self.con().cursor()
        c.execute("""
            SELECT id, jif, length
            FROM papers""")
        for paper_id, jif, length in c:
            paper_jifs[paper_id] = jif if jif is not None else 0
            paper_lengths[paper_id] = length

        # Median length for papers with 0 length = no length available
        paper_lengths[np.isnan(paper_lengths)] =\
            np.median(paper_lengths[~np.isnan(paper_lengths)])

        # For #coauthors
        print("Loading paper authors...")
        paper_authors = cbor2.load(open(
            os.path.join(settings.DATA_DIR, 'arxiv', 'keywords-backend',
                         'nn_data', 'paper_authors.cbor'),
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
        tmp = list(set([category.split('.')[0]
                        for category in categories]))
        # Ensure deterministic category ids, which is prevented by set()
        tmp.sort()
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

        # Per-paper input data
        # Prepare for sparse array with shape (ntotalpapers, nfeatures)
        print("Preparing sparse rows/cols/data...")
        rows = np.concatenate([i*np.ones(sparse_xcolumns)
                               for i in range(0, ntotalpapers)])
        # NOTE: The column is not correct for the 'category', which is
        # corrected in the loop below
        cols = np.tile(np.arange(sparse_xcolumns), ntotalpapers)
        data = np.zeros(rows.shape)

        datapos = 0
        for paper_id in range(0, ntotalpapers):
            tmp = np.zeros(sparse_xcolumns)

            # Explicitly tell net which data are padding and which are papers
            index = self.data_positions['padding']
            tmp[index] = 1

            # num_authors
            index = self.data_positions['num_authors']
            tmp[index] = len(paper_authors[paper_id])

            # months
            index = self.data_positions['months']
            tmp[index] = (self.get_cutoff_date() -
                          date.fromtimestamp(paper_dates[paper_id])).days/30

            # length
            index = self.data_positions['length']
            tmp[index] = paper_lengths[paper_id]

            # num_citations
            index = self.data_positions['num_citations']
            tmp[index] = paper_citations[paper_id]

            # published
            # TODO: Add 'published'

            # jif
            index = self.data_positions['jif']
            tmp[index] = paper_jifs[paper_id]

            # categories
            # NOTE: Categories are saved as a one-hot vector in the sparse
            # matrix. However, we only save the '1' and not the '0's of this
            # one-hot vector to save memory. Therefore, we must adjust the
            # column to save this '1' to
            index = self.data_positions['categories']
            # The value to save is '1'
            # After flattening, this '1' ends up at
            # tmp.flatten()[paperi*tmp.shape[1] + index]
            tmp[index] = 1
            # The column to save this to is the category_index'th category
            # column (which start at 'index')
            category_index = categories[paper_categories[paper_id]]
            cols[datapos + index] = index + category_index

            # Translate to sparse matrix format
            tmp = tmp.flatten()
            data[datapos:datapos+len(tmp)] = tmp
            datapos += len(tmp)

        # Consistency check
        if datapos != len(data):
            raise Exception("datapos != len(data)")

        # Free memory
        print("Freeing memory...")
        del paper_authors
        del paper_dates
        del paper_categories
        del tmp

        import gc
        gc.collect()

        # Create sparse matrix for paperx
        paperx = sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(ntotalpapers, len(xcolumns)))

        return paperx

    def load_data_paperx(self):
        try:
            file_paperx = os.path.join(
                self.data_dir_xy,
                'net_data_paperx%s.npz' % ('-future' if self._future else ''))
            paperx = sparse.load_npz(file_paperx)
        except FileNotFoundError as _:
            paperx = self._generate_paper_x()
            sparse.save_npz(file_paperx, paperx)
        return paperx

    def generate_data_x(self):

        print("Generating x data for authors...")
        paperx = self.load_data_paperx()

        # Load author->papers mapping
        author_papers = self._load_author_papers()
        effective_max_papers = len(max(author_papers, key=lambda x: len(x)))
        if effective_max_papers != 1273:
            # TODO
            raise Exception("Note to self: Modify also self.load_data_x")
        print("Effective max papers", effective_max_papers)

        # Load paper_dates to throw out papers before cutoff
        paper_dates = self._load_paper_dates()

        cutoff_date = mktime(self.get_cutoff_date().timetuple())
        print("The cutoff is", self.get_cutoff_date())

        print("Removing papers before cutoff...")
        author_papers = [
            [paper_id for paper_id in papers
             if paper_dates[paper_id] < cutoff_date]
            for papers in author_papers]

        author_ids = [author_id
                      for _, author_id in self.get_data_author_generator()]
        numauthors = len(author_ids)
        print("#authors", numauthors)

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

        ntotalpapers = 0
        for author_id in author_ids:
            ntotalpapers += len(author_papers[author_id])

        # Per-paper input data
        # Prepare for sparse array with shape (nauthors*npapers, nfeatures)
        print("Preparing sparse rows/cols/data...")
        rows = np.zeros(ntotalpapers*sparse_xcolumns, dtype=np.int64)
        pos = 0
        for i, author_id in enumerate(author_ids):
            numpapers = len(author_papers[author_id])
            if not numpapers:
                continue
            tmp = np.concatenate(
                [i*effective_max_papers + j*np.ones(sparse_xcolumns)
                 for j in range(0, numpapers)])
            rows[pos:pos+len(tmp)] = tmp
            pos += len(tmp)
        if pos != rows.shape[0]:
            raise Exception("Invalid pos?!")
        # NOTE: The column is not correct for the 'category', which is
        # corrected in the loop below
        cols = np.tile(np.arange(sparse_xcolumns), ntotalpapers)
        data = np.zeros(rows.shape)

        # Per-author input data
        xaux = np.zeros((numauthors, len(xauxcolumns)))

        datapos = 0
        for i, author_id in enumerate(author_ids):
            papers = author_papers[author_id]
            if not len(papers):
                continue
            print(author_id, "has #papers:", len(papers))

            # Take data from paperx. This contains zeros from one-hot category
            # vector
            tmp = paperx[papers].toarray().reshape(
                (len(papers), len(xcolumns)))

            # Find category indices by looking at where the ones are
            index = self.data_positions['categories']
            category_indices = np.where(
                tmp[:, index:index+self.numcategories] == 1)[1]

            # Remove the zeros for our final sparse format
            tmp = np.delete(tmp, np.s_[index:index+self.numcategories-1],
                            axis=1)
            tmp[:, index] = np.ones(len(papers))
            assert(tmp.shape == (len(papers), sparse_xcolumns))

            # categories
            # NOTE: Categories are saved as a one-hot vector in the sparse
            # matrix. However, we only save the '1' and not the '0's of this
            # one-hot vector to save memory. Therefore, we must adjust the
            # column to save this '1' to
            for paperi, paper in enumerate(papers):
                # The value to save is '1'
                # After flattening, this '1' ends up at
                # tmp.flatten()[paperi*tmp.shape[1] + index]
                assert(tmp[paperi, index] == 1)
                # The column to save this to is the category_index'th category
                # column (which start at 'index')
                category_index = category_indices[paperi]
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
        file_x = os.path.join(
            self.data_dir_xy,
            'net_data_x%s.npz' % ('-future' if self._future else ''))
        file_xaux = os.path.join(
            self.data_dir_xy,
            'net_data_xaux%s.h5' % ('-future' if self._future else ''))

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
            # TODO: Don't hardcode 1273
            npapers = 1273
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

                    # NOTE: y contains at 0 position the h-index at the cutoff
                    # But the net only predicts differences for the future.
                    yield ({'perpaper_inputs': xbatch,
                            'perauthor_inputs': xaux[start:end]},
                            y[start:end, 1:])
                    start += author_batch_size
                    end += author_batch_size

        return generator, batches_per_epoch, nauthors, npapers, nfeatures

    def train(self, activation='tanh', load=False):

        # Speed up batch evaluations
        self.clear_keras_session()

        # Load data
        print("Loading net training data...")
        x, xaux, y = self.load_train_data()

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

            outputs = Dense(units=self.predict_after_years,
                            activation='relu')(tmp)

            model = Model(inputs=[perpaper_inputs, perauthor_inputs],
                          outputs=outputs)

            model.compile(
                loss='mean_squared_error',
                optimizer='adam')

        # Fit keras model
        model.fit_generator(generator(), epochs=self.epochs,
                            steps_per_epoch=batches_per_epoch)
        model.save(self.get_net_filename())

    def predict_no_cumsum(self, x, xaux, y):
        # Speed up batch evaluations
        self.clear_keras_session()

        # Prepare generator
        generator, batches_per_epoch, *_ = \
            self.get_generator(x, xaux, y)

        print("Predicting...")
        y_net = self.load_model().predict_generator(
            generator(), steps=batches_per_epoch)

        # NOTE: Does not do np.cumsum(y_net, axis=1). This is already done in
        # self.do_evaluate()

        # NOTE: Add h0 at cutoff to 0 prediction since we here only predict
        # differences with the scimeter net, while the Net, TimeSeriesNet
        # expect an abolute h-index for the first prediction!
        y_net[:, 0] += y[:, 0]

        return y_net

    def evaluate(self):
        filename = self.get_net_filename().replace('.h5',
                                                   '-validation-results.npz')
        try:
            tmp = np.load(filename)
            y = tmp['y_true']
            y_net = tmp['y_pred']
        except FileNotFoundError:
            # Load data
            print("Loading net evaluation data...")
            x, xaux, y = self.load_validation_data()
            y_net = self.predict_no_cumsum(x, xaux, y)
            np.savez(filename, y_true=y, y_pred=y_net)

        # NOTE: do_evaluate() expecteds y to start at cutoff+1 years. But here
        # we start at cutoff. So add the h0 at cutoff to the diff at +1 years.
        # Then we can use y[:, 1] for do_evaluate()
        y[:, 1] += y[:, 0]

        self.metric_mapemin5 = True
        result = self.do_evaluate(y[:, 1:], y_net)
        self.metric_mapemin5 = False
        return result

    def predict_all(self):
        # Load data
        print("Loading all data...")
        # Load all authors, not just validation authors
        indices = np.arange(len(self.get_train_authors()))
        x, xaux, y = self.load_validation_data(indices=(indices,))

        y_net = self.predict_no_cumsum(x, xaux, y)
        if self.force_monotonic:
            y_net = np.cumsum(y_net, axis=1)

        filename = self.get_net_filename().replace(
            '.h5', '%s_all.npz' % ('-future' if self._future else '-predict'))
        if self._future:
            np.savez(filename, y_pred=y_net)
        else:
            np.savez(filename, y_pred=y_net, y_true=y)

    def future_predict_all(self):
        self._future = True
        self.predict_all()
        self._future = False

    def future_predict_by_papers(self):

        # Must be set before selecting papers so we have the correct cutoff
        # date
        self._future = True

        # TODO: For now just check author_id = 401871
        cutoff_date = mktime(self.get_cutoff_date().timetuple())
        c = self.con().cursor()
        c.execute("""
            SELECT p.id FROM papers AS p
            INNER JOIN author_papers AS pa
            ON p.id = pa.paper_id
            WHERE
            pa.author_id = 401871 AND
            p.timestamp < %(cutoff_date)s""", {
                'cutoff_date': cutoff_date})
        tmp = []
        for (paper_id,) in c:
            tmp.append(paper_id)
        by_papers = [tmp]

        # Load model
        model = self.load_model()
        input_shape = model.get_layer(index=0).input_shape
        nfeatures = input_shape[2]
        npapers = input_shape[1]

        # Generate input x, xaux
        paperx = self.load_data_paperx()

        # Calculate h-index at cutoff
        hindex_at_cutoff = np.zeros(len(by_papers))
        for i, papers in enumerate(by_papers):
            ncits = paperx[papers, self.data_positions['num_citations']].toarray().flatten()
            ncits[::-1].sort()

            tmp = 0
            for ncit in ncits:
                if ncit <= tmp:
                    break
                tmp += 1
            hindex_at_cutoff[i] = tmp

        # Scale x
        self._scale_inputs(paperx, xaux=None, is_train_inputs=False)

        # NOTE: In principle here one could start a loop and calculate
        # for different by_papers arrays very quickly as everything is already
        # set up

        # Generate input data
        x = np.zeros((len(by_papers), npapers, nfeatures))
        for i, papers in enumerate(by_papers):
            x[i, :len(papers), :] = paperx[papers].toarray()
        xaux = np.zeros((len(by_papers), 0))

        # Predict
        y_net = model.predict({'perpaper_inputs': x, 'perauthor_inputs': xaux})
        # Add hindex at cutoff since the network only predicts differences
        y_net[:, 0] += hindex_at_cutoff
        if self.force_monotonic:
            y_net = np.cumsum(y_net, axis=1)

        self._future = False

        print(y_net)
        return y_net


if __name__ == '__main__':

    import argparse
    from runner import *

    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['train', 'evaluate', 'future-all',
                                           'future-by-papers', 'predict-all'])

    args = parser.parse_args()
    n = ScimeterNet()
    if args.action == 'train':
        n.train()
    elif args.action == 'evaluate':
        n.evaluate()
    elif args.action == 'future-all':
        n.future_predict_all()
    elif args.action == 'future-by-papers':
        n.future_predict_by_papers()
    elif args.action == 'predict-all':
        n.predict_all()
    else:
        raise Exception("Invalid action")
