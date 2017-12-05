from analysis import Analysis
from db import db
import settings
import os
from dateutil.relativedelta import relativedelta
import pickle
import numpy as np
import pandas as pd


class Net(Analysis):

    epochs = 150

    data_positions = {
        'padding': 0,
        'num_citations': 1,
        'months': 2,
        'pagerank': 3,
        'length': 4,
        'published': 5,
        'jif': 6,
        'num_coauthors': 7,
        'avg_coauthor_pagerank': 8,
        'max_coauthor_pagerank': 9,
        'min_coauthor_pagerank': 10,
        'categories': 11,
        'paper_topics': 11+38
        }
    data_positions_aux = {
        'broadness_lda': 0,
        }
    paper_topics_dim = 50
    numcategories = 38

    ignore_max_hindex_before = True

    def __init__(self, cutoff, target, first_paper_range=None):

        super().__init__(cutoff, first_paper_range)

        # Create dir for net files if it doesn't exist
        os.makedirs(os.path.join(self.data_dir, 'models'), exist_ok=True)

        # Directory for net x and y data may be set to a different location
        # (e.g. for running on cluster)
        self.data_dir_xy = self.data_dir

        if target == 'hindex_cumulative':
            target_table = 'analysis{0}_hindex_data'.format(self.suffix_cuts)
            target_field = 'hindex_cumulative'

            def target_func(x):
                return x
        elif target == 'sqrt_nc_after':
            target_table = 'analysis{0}_nc_data'.format(self.suffix_cuts)
            target_field = 'nc_after'
            target_func = np.sqrt
        else:
            raise Exception("Invalid target")

        # Adjust epochs
        if self.first_paper_range == (10, 14):
            self.epochs = 340
        elif self.first_paper_range == (5, 12):
            self.epochs = 150
        else:
            raise Exception("Not yet implemented")

        self.target_func = target_func
        self.target_table = target_table
        self.target_field = target_field

        self.target = target
        self.suffix = self.target
        # Used by cross-validation
        self.suffix_author_ids = ""

        self.__exclude_data = []
        self.__exclude_data_aux = []

    def __add_column(self, columns, field, i, exclude):
        if not exclude or field not in self.__exclude_data:
            columns.append("%s%s" % (field, i))

    def __get_columns(self, numfeatures, data_positions, exclude):
        columns = []
        last_field = None
        last_pos = None
        for field, pos in sorted(data_positions.items(),
                                 key=lambda x: x[1]):
            if field != last_field and last_field is not None:
                for i in range(0, pos - last_pos - 1):
                    self.__add_column(columns, last_field, i+1, exclude)
            self.__add_column(columns, field, 0, exclude)

            last_field = field
            last_pos = pos
        final_pos = numfeatures - 1
        for i in range(0, final_pos - last_pos):
            self.__add_column(columns, last_field, i+1, exclude)

        return columns

    def get_column_names_x(self, exclude=False):

        numfeatures = len(self.data_positions)
        numfeatures += (self.numcategories-1) + (self.paper_topics_dim-1)

        return self.__get_columns(numfeatures, self.data_positions, exclude)

    def get_column_names_xaux(self, exclude=False):
        numfeatures = len(self.data_positions_aux)

        return self.__get_columns(numfeatures, self.data_positions_aux,
                                  exclude)

    def set_exclude_data(self, exclude_data):
        print('set_exclude_data', exclude_data)
        self.__exclude_data = exclude_data

    def choose_train_authors(self):
        c = db().cursor()

        # Check if already selected
        c.execute("""
            SELECT COUNT(*) FROM analysis{0}_authors
            WHERE train IS NULL
            """.format(self.suffix_cuts))
        numauthors = c.fetchone()[0]
        if numauthors == 0:
            print("Already selected")
            return

        # Take ~70% training
        if self.first_paper_range == (10, 14):
            numtraining = 13500
        else:
            numtraining = int(np.ceil(numauthors * .7 / 500.) * 500.)

        c.execute("""
            UPDATE analysis{0}_authors
            SET train = 1
            ORDER BY RAND()
            LIMIT %(numtraining)s
            """.format(self.suffix_cuts), {
                'numtraining': numtraining})
        c.execute("""
            UPDATE analysis{0}_authors
            SET train = 0
            WHERE train IS NULL
            """.format(self.suffix_cuts))
        db().commit()

        # Save train / validation authors to file
        self.save_train_authors()

    def get_author_ids_filename(self):
        return os.path.join(self.data_dir, 'author_ids%s.npy' %
                            self.suffix_author_ids)

    def save_train_authors(self):
        c = db().cursor()

        max_hindex_before, _ = self.get_hindex_predictor()
        numauthors = c.execute("""
            SELECT
            a.author_id,
            IF(
                a.train = 1,
                1,
                IF(
                    a.train = 0 AND h.hindex_before <= %(max_hindex_before)s,
                    0,
                    -1
                )
            )

            FROM analysis{0}_authors AS a

            INNER JOIN analysis{0}_hindex_data AS h
            ON h.author_id = a.author_id

            WHERE h.predict_after_years = %(predict_after_years)s

            ORDER BY a.author_id ASC""".format(self.suffix_cuts), {
                'predict_after_years': self.predict_after_years,
                'max_hindex_before': max_hindex_before})
        np.save(self.get_author_ids_filename(),
                np.fromiter(c, count=numauthors,
                            dtype=[('author_id', 'i4'), ('train', 'i4')]))

    def get_train_authors(self, generate_if_not_exists=True):
        try:
            return np.load(self.get_author_ids_filename())
        except FileNotFoundError as e:
            if generate_if_not_exists:
                self.save_train_authors()
                return np.load(self.get_author_ids_filename())
            else:
                raise e

    def get_effective_max_papers(self):
        try:
            return self.__effective_max_papers
        except AttributeError as e:
            c = db().cursor()
            if self.cutoff == self.CUTOFF_PERAUTHOR:
                c.execute("""
                    SELECT COUNT(*)
                    FROM `analysis{0}_fast_paper_authors` AS pa
                    INNER JOIN analysis{0}_authors AS a
                    ON a.author_id = pa.author_id
                    WHERE pa.date_created <= DATE_ADD(
                            a.first_paper_date,
                            INTERVAL %(split_after_years)s YEAR)
                    GROUP BY pa.author_id
                    ORDER BY `COUNT(*)` DESC
                    LIMIT 1""".format(self.suffix_cuts), {
                        'split_after_years': self.split_after_years})
            elif self.cutoff == self.CUTOFF_SINGLE:
                c.execute("""
                    SELECT COUNT(*)
                    FROM `analysis{0}_fast_paper_authors` AS pa
                    INNER JOIN analysis{0}_authors AS a
                    ON a.author_id = pa.author_id
                    WHERE pa.date_created <= %(cutoff_date)s
                    GROUP BY pa.author_id
                    ORDER BY `COUNT(*)` DESC
                    LIMIT 1""".format(self.suffix_cuts), {
                        'cutoff_date': self.cutoff_date})
            else:
                raise Exception("Not implemented")
            self.__effective_max_papers = c.fetchone()[0]
            return self.__effective_max_papers

    def __get_categories(self):
        try:
            return self.__categories_data
        except AttributeError:
            # For categories
            # NOTE: Taking all categories (172) takes too much memory
            # Therefore, reduce A.* to A which leaves only 38 categories
            c = db().cursor()
            c.execute("""SELECT id, name FROM categories ORDER BY name ASC""")

            categories = {}
            numcategories = 0
            last_coarse_name = None
            for row in c:
                id = row[0]
                name = row[1]
                coarse_name = name.split('.')[0]

                if coarse_name != last_coarse_name:
                    numcategories += 1

                categories[id] = numcategories-1
                last_coarse_name = coarse_name

            self.__categories_data = numcategories, categories
            return self.__categories_data

    def __load_paper_topics(self):
        print("Loading paper topics...")
        backend_papers = pickle.load(open(
            os.path.join(settings.DATA_DIR,
                         'arxiv', 'keywords-backend', 'papers'),
            'rb'
        ))
        backend_papers = {v: k for k, v in enumerate(backend_papers)}

        paper_topics = np.load(open(
            os.path.join(settings.DATA_DIR,
                         'arxiv', 'keywords-backend', 'paper_topics'),
            'rb'
        ))
        if self.paper_topics_dim != paper_topics.shape[1]:
            raise Exception("self.paper_topics_dim is wrong!")

        c = db().cursor()
        if self.cutoff == self.CUTOFF_SINGLE:
            c.execute("""
                SELECT id, arxiv_id FROM papers
                WHERE date_created < %(cutoff_date)s""", {
                    'cutoff_date': self.cutoff_date})
        else:
            c.execute("""SELECT id, arxiv_id FROM papers""")

        return {row[0]: paper_topics[backend_papers[row[1]]] for row in c}

    def get_data_author_generator(self, max_hindex_before, train=None,
                                  author_ids=None):

        join_hindex = False
        join_nc = False

        if max_hindex_before:
            join_hindex = True
        if self.target_table == \
                'analysis{0}_hindex_data'.format(self.suffix_cuts):
            join_hindex = True
            target_alias = 'hindex'
        elif self.target_table == \
                'analysis{0}_nc_data'.format(self.suffix_cuts):
            join_nc = True
            target_alias = 'nc'

        sql = """SELECT
                    a.author_id,
                    a.first_paper_date,
                    %s.%s,
                    a.broadness_lda
                FROM analysis{0}_authors AS a""".format(self.suffix_cuts) % \
              (target_alias, self.target_field)
        if join_hindex:
            sql += """
                INNER JOIN analysis{0}_hindex_data AS hindex
                ON hindex.author_id = a.author_id AND
                hindex.predict_after_years = %(predict_after_years)s
                """.format(self.suffix_cuts)
        if join_nc:
            sql += """
                INNER JOIN analysis{0}_nc_data AS nc
                ON nc.author_id = a.author_id AND
                nc.predict_after_years = %(predict_after_years)s
                """.format(self.suffix_cuts)
        sql += "WHERE 1=1"
        if train is not None:
            sql += """
                AND a.train = %(train)s"""
        if max_hindex_before:
            sql += """
                AND hindex.hindex_before <= %(max_hindex_before)s"""
        if author_ids is not None:
            sql += """
                AND a.author_id IN(%s)""" \
                    % ', '.join([str(id) for id in author_ids])
        sql += """
            ORDER BY a.author_id ASC"""

        c = db().cursor()
        numauthors = c.execute(sql, {
                'predict_after_years': self.predict_after_years,
                'train': train,
                'max_hindex_before': max_hindex_before})

        def generator():
            for i, row in enumerate(c):
                yield i, row

        if not numauthors:
            raise Exception("No authors found - probably nc/hindex_data not"
                            "generated?")

        return numauthors, generator

    def generate_data_x(self, train=None, max_hindex_before=None,
                        author_generator=None):

        print("Generating x data...")

        c = db().cursor()
        c2 = db().cursor()

        # For paper topics
        paper_topics = self.__load_paper_topics()
        paper_topics_dim = paper_topics[min(paper_topics)].shape[0]
        if paper_topics_dim != self.paper_topics_dim:
            raise Exception("paper_topics_dim != self.paper_topics_dim")

        # For papers with no length take median paper length
        # https://stackoverflow.com/a/7263925
        c.execute("""
            SELECT avg(t1.length) as median_val FROM (
                SELECT @rownum:=@rownum+1 as `row_number`, d.length
                FROM papers d,  (SELECT @rownum:=0) r
                WHERE 1
                ORDER BY d.length
            ) as t1,
            (
                SELECT count(*) as total_rows
                FROM papers d
                WHERE 1
            ) as t2
            WHERE 1
            AND t1.row_number
            IN ( floor((total_rows+1)/2), floor((total_rows+2)/2) )
            """)
        median_length = c.fetchone()[0]

        # For padding
        effective_max_papers = self.get_effective_max_papers()

        # For categories
        numcategories, categories = self.__get_categories()
        if numcategories != self.numcategories:
            raise Exception("numcategories != self.numcategories")

        # Get author generator
        if author_generator:
            numauthors, author_generator = author_generator
        else:
            numauthors, author_generator = self. \
                get_data_author_generator(max_hindex_before, train=train)

        xcolumns = self.get_column_names_x(exclude=False)
        xauxcolumns = self.get_column_names_xaux(exclude=False)

        # Per-paper input data
        x = np.zeros((numauthors, effective_max_papers, len(xcolumns)))
        # Per-author input data
        xaux = np.zeros((numauthors, len(xauxcolumns)))

        for i, row in author_generator():
            author_id = row[0]
            first_paper_date = row[1]
            broadness_lda = row[3]

            print(author_id, first_paper_date)

            # First per-author input data
            index = self.data_positions_aux['broadness_lda']
            xaux[i][index] = broadness_lda

            # Then per-paper input data

            # First fetch list of all papers with pagerank, date
            # and citations counts
            if self.cutoff == self.CUTOFF_SINGLE:
                months_since_date = self.cutoff_date
            elif self.cutoff == self.CUTOFF_PERAUTHOR:
                months_since_date = first_paper_date
            else:
                raise Exception("Not implemented")
            sql = """
                SELECT
                    pa.paper_id,
                    DATEDIFF(pa.date_created, %(months_since_date)s)/30
                        AS months,
                    pa.pagerank,
                    IFNULL(pa.length, %(median_length)s),
                    pa.published,
                    pa.jif,
                    IF(c.cited_paper IS NULL, 0, COUNT(*)) AS num_citations
                FROM analysis{0}_fast_paper_authors AS pa

                LEFT JOIN analysis{0}_fast_citations AS c
                ON c.cited_paper = pa.paper_id
                    AND c.citing_paper_date_created >= %(start_date)s
                    AND c.citing_paper_date_created < %(end_date)s

                WHERE
                    pa.author_id = %(author_id)s
                    AND pa.date_created >= %(start_date)s
                    AND pa.date_created < %(end_date)s

                GROUP BY pa.paper_id
                ORDER BY num_citations DESC

                LIMIT %(effective_max_papers)s
                """.format(self.suffix_cuts)
            numpapers = c2.execute(sql, {
                'author_id': author_id,
                'months_since_date': months_since_date,
                'median_length': median_length,
                'start_date': first_paper_date,
                'end_date': self.get_split_date(first_paper_date),
                'effective_max_papers': effective_max_papers})

            paper_data = np.fromiter(
                c2, count=numpapers, dtype=[
                    ('paper_id', 'i4'),
                    ('months', 'f4'),
                    ('pagerank', 'f4'),
                    ('length', 'f4'),
                    ('published', 'f4'),
                    ('jif', 'f4'),
                    ('num_citations', 'f4'),
                    ])
            paper_ids_str = ','.join([str(e) for e in paper_data['paper_id']])
            paper_indices = {
                paper_id: i
                for (i, paper_id) in enumerate(paper_data['paper_id'])}

            # Explicitely tell net which data are padding and which are papers
            index = self.data_positions['padding']
            x[i, :len(paper_data), index] = np.ones(len(paper_data))

            # Put num_citations in x
            index = self.data_positions['num_citations']
            x[i, :len(paper_data), index] = paper_data['num_citations']
            index += 1

            # Put months since first_paper_date in x
            index = self.data_positions['months']
            x[i, :len(paper_data), index] = paper_data['months']

            # Put pagerank in x
            index = self.data_positions['pagerank']
            x[i, :len(paper_data), index] = paper_data['pagerank']

            # Put length in x
            index = self.data_positions['length']
            x[i, :len(paper_data), index] = paper_data['length']

            # Put published in x
            index = self.data_positions['published']
            x[i, :len(paper_data), index] = paper_data['published']

            # Put JIF in x
            index = self.data_positions['jif']
            x[i, :len(paper_data), index] = paper_data['jif']

            # Get coauthors of papers before split time
            # NOTE: Also count authors not in analysis_fast_paper_authors!
            sql = """
                SELECT
                    COUNT(*)-1 AS num_coauthors,
                    AVG(coa.coauthor_pagerank) AS avg_coauthor_pagerank,
                    MAX(coa.coauthor_pagerank) AS max_coauthor_pagerank,
                    MIN(coa.coauthor_pagerank) AS min_coauthor_pagerank

                FROM paper_authors AS pa

                LEFT JOIN analysis{0}_fast_coauthors AS coa
                ON coa.coauthor_id = pa.author_id
                    AND coa.analysis_author_id = %(author_id)s

                WHERE
                pa.paper_id IN({1})

                GROUP BY pa.paper_id
                ORDER BY FIELD(pa.paper_id, {1})
                """.format(self.suffix_cuts, paper_ids_str)
            numpapers = c2.execute(sql, {
                'author_id': author_id})

            coauthors_data = np.fromiter(
                c2, count=numpapers, dtype=[
                    ('num_coauthors', 'i4'),
                    ('avg_coauthor_pagerank', 'f4'),
                    ('max_coauthor_pagerank', 'f4'),
                    ('min_coauthor_pagerank', 'f4')]
                )

            # Consistency check
            if len(coauthors_data) != len(paper_data):
                raise Exception("lengths don't match?!")

            # Put coauthor data in x
            index = self.data_positions['num_coauthors']
            x[i, :len(coauthors_data), index] = \
                coauthors_data['num_coauthors']
            index = self.data_positions['avg_coauthor_pagerank']
            x[i, :len(coauthors_data), index] = \
                np.nan_to_num(coauthors_data['avg_coauthor_pagerank'])
            index = self.data_positions['max_coauthor_pagerank']
            x[i, :len(coauthors_data), index] = \
                np.nan_to_num(coauthors_data['max_coauthor_pagerank'])
            index = self.data_positions['min_coauthor_pagerank']
            x[i, :len(coauthors_data), index] = \
                np.nan_to_num(coauthors_data['min_coauthor_pagerank'])

            # Get categories
            # NOTE: This is actually very expensive since
            # effective_max_papers*numcategories ~ 35000
            # compared to the other data 2*effective_max_papers ~ 400
            sql = """
                SELECT pc.paper_id, pc.category_id

                FROM paper_categories AS pc

                WHERE
                pc.paper_id IN("""+paper_ids_str+""")

                GROUP BY pc.paper_id, pc.category_id
                ORDER BY FIELD(pc.paper_id, """+paper_ids_str+""")
                """
            c2.execute(sql)

            last_paper = None
            index = self.data_positions['categories']
            for row in c2:
                paper_index = paper_indices[row[0]]
                category_index = categories[row[1]]
                x[i][paper_index][index + category_index] = 1

            # Add paper topics
            index = self.data_positions['paper_topics']
            for paper_id, paper_index in paper_indices.items():
                x[i][paper_index][index:index+paper_topics_dim] = \
                    paper_topics[paper_id]

            # Consistency check
            if index != 11 + numcategories:
                raise Exception("index != 11 + numcategories")

        # xaux to DataFrame
        xaux = pd.DataFrame(xaux, columns=xauxcolumns)

        # x to DataFrame
        # We have (nauthor, npaper, nfeature), for DataFrame reshape to
        # Use (nauthor * npaper, nfeature)
        x = pd.DataFrame(x.reshape(x.shape[0] * x.shape[1], x.shape[2]),
                         columns=xcolumns)

        return x, xaux

    def generate_data_y(self, train=None, max_hindex_before=None,
                        author_generator=None):

        print("Generating y data...")

        # Get author generator
        if author_generator:
            numauthors, author_generator = author_generator
        else:
            numauthors, author_generator = self. \
                get_data_author_generator(max_hindex_before, train=train)

        # Output data
        y = np.zeros(numauthors)

        for i, row in author_generator():
            author_id = row[0]
            first_paper_date = row[1]
            target = row[2]

            print(author_id, first_paper_date, target)

            # Store y value
            y[i] = self.target_func(target)

        return pd.DataFrame(y, columns=['y10'])

    def get_xindices(self, file_x, indices, total_authors):

        if indices is None:
            return None

        # self.get_effective_max_papers() not allowed since is uses the DB
        # Therefore, calculate it from total_authors and nrows
        if total_authors is None:
            raise ValueError("total_authors needs to be set")

        with pd.HDFStore(file_x, 'r') as store:
            nrows = store.get_storer('x').nrows
        numpapers = int(nrows / total_authors)

        if nrows % total_authors != 0:
            raise Exception("nrows / total_authors not an integer?!")

        tmp = np.array([i for i in range(0, numpapers)])
        xindices = np.zeros(numpapers * len(indices), dtype=int)
        index = 0
        for author_index in indices:
            xindices[index:index+numpapers] = \
                numpapers * author_index + tmp
            index += numpapers

        return xindices

    def load_data_x(self, indices=None, total_authors=None):
        file_x = os.path.join(self.data_dir_xy, 'net_data_x.h5')
        file_xaux = os.path.join(self.data_dir_xy, 'net_data_xaux.h5')

        if indices is not None:
            indices = indices[0]

        # Columns to load
        xcolumns = self.get_column_names_x(exclude=True)
        xauxcolumns = self.get_column_names_xaux(exclude=True)

        try:
            xaux = pd.read_hdf(file_xaux, where=pd.IndexSlice[indices],
                               columns=xauxcolumns)

            xindices = self.get_xindices(file_x, indices, total_authors)
            x = pd.read_hdf(file_x, where=pd.IndexSlice[xindices],
                            columns=xcolumns)
        except (FileNotFoundError, KeyError) as e:
            x, xaux = self.generate_data_x()
            xaux.to_hdf(file_xaux, key='xaux', format='table')
            x.to_hdf(file_x, key='x', format='table')
            if indices is None:
                xaux = xaux[xauxcolumns]
                x = x[xcolumns]
            else:
                xaux = xaux.iloc[indices][xauxcolumns]

                xindices = self.get_xindices(file_x, indices, total_authors)
                x = x.iloc[xindices][xcolumns]

        return x, xaux

    def load_data_y(self, indices=None):
        file_y = os.path.join(self.data_dir_xy,
                              'net_data_y-%s-%s.h5' %
                              (self.predict_after_years, self.target))

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

    def load_train_data(self, indices=None):

        # Pick training authors
        authors = self.get_train_authors()
        if indices is None:
            indices = np.where(authors['train'] == 1)

        x, xaux = self.load_data_x(indices=indices, total_authors=len(authors))
        y = self.load_data_y(indices=indices)

        # Scale data
        self.__scale_inputs(x, xaux, is_train_inputs=True)

        # Reshape x to (nauthos, npaper, nfeature)
        x = x.values.reshape((len(indices[0]), -1, x.values.shape[1]))

        return x, xaux.values, y.values

    def pick_validation_authors(self, authors):
        if self.ignore_max_hindex_before:
            return np.where(authors['train'] <= 0)
        else:
            return np.where(authors['train'] == 0)

    def load_validation_data_y(self, indices=None):
        return self.load_validation_data(only_y=True, indices=indices)

    def load_validation_data(self, only_y=False, indices=None):

        # Pick training authors
        authors = self.get_train_authors()
        if indices is None:
            indices = self.pick_validation_authors(authors)

        y = self.load_data_y(indices=indices)
        if only_y:
            return y.values

        x, xaux = self.load_data_x(indices=indices,
                                   total_authors=len(authors))

        # Scale data
        self.__scale_inputs(x, xaux, is_train_inputs=False)

        # Reshape x to (nauthos, npaper, nfeature)
        x = x.values.reshape((len(indices[0]), -1, x.values.shape[1]))

        return x, xaux.values, y.values

    def load_validation_inputs(self, *args, **kwargs):
        x, xaux, y = self.load_validation_data(*args, **kwargs)
        inputs = {'perpaper_inputs': x, 'perauthor_inputs': xaux}
        return inputs, y

    def __scale_inputs(self, x, xaux, is_train_inputs):

        from sklearn.preprocessing import StandardScaler
        from sklearn.externals import joblib

        scale_fields = [
            'padding',
            'num_citations',
            'months',
            'pagerank',
            'length',
            'published',
            'jif',
            'num_coauthors',
            'avg_coauthor_pagerank',
            'max_coauthor_pagerank',
            'min_coauthor_pagerank',
            # 'categories',
            # 'paper_topics'
            ]

        for field in scale_fields:
            column = field + "0"
            if column not in x.columns and field in self.__exclude_data:
                continue

            filename = os.path.join(self.data_dir,
                                    'scaler-%s' % field)
            try:
                scaler = joblib.load(filename)
            except FileNotFoundError:
                if is_train_inputs:
                    scaler = StandardScaler(copy=False)
                    scaler.fit(x[[column]])
                    joblib.dump(scaler, filename)
                else:
                    raise Exception("Scaler not found")

            x[[column]] = scaler.transform(x[[column]])

        scale_fields_aux = [
            'broadness_lda',
            ]
        for field in scale_fields_aux:
            column = field + "0"
            if column not in x.columns and field in self.__exclude_data:
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

    def clear_keras_session(self):
        # This is called before each train / evaluate because otherwise things
        # get slow over time when batch training/evaluating from runner.py
        # https://stackoverflow.com/questions/45063602/attempting-to-reset-tensorflow-graph-when-using-keras-failing
        import keras.backend as K
        K.clear_session()

    def get_net_filename(self):
        return os.path.join(self.data_dir, 'models', 'net-%s.h5' % self.suffix)

    def load_model(self):
        # Load keras model
        print("Loading keras...")
        from keras.models import load_model

        print("Loading model...")
        return load_model(self.get_net_filename())

    def train(self, activation='tanh', live_validate=False, load=False):

        # Load validation data
        if live_validate:
            validation_data = self.load_validation_inputs()
        else:
            validation_data = None

        # Speed up batch evaluations
        self.clear_keras_session()

        # Load data
        print("Loading net training data...")
        x, xaux, y = self.load_train_data()

        # Build/load keras model
        if load:
            print("Loading keras / load model...")
            model = self.load_model()
        else:
            print("Loading keras / build model...")
            from keras.models import Model
            from keras.layers import Input, Dense, Conv1D, concatenate, \
                Dropout, GlobalAveragePooling1D

            perpaper_inputs = Input(shape=x[0].shape,
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

            # tmp = Dropout(.2)(tmp)

            tmp = concatenate([tmp, perauthor_inputs])
            tmp = Dense(units=70, activation=activation)(tmp)

            outputs = Dense(units=1, activation='relu')(tmp)
            lossfunction = 'mean_squared_error'

            model = Model(inputs=[perpaper_inputs, perauthor_inputs],
                          outputs=outputs)

            model.compile(
                loss=lossfunction,
                optimizer='adam')

        # Fit keras model
        model.fit({'perpaper_inputs': x, 'perauthor_inputs': xaux}, y,
                  epochs=self.epochs, batch_size=50,
                  validation_data=validation_data)
        model.save(self.get_net_filename())

    def get_hindex_predictor(self, all_years=False):

        c = db().cursor()

        join_nc = False
        if self.target_table == \
                'analysis{0}_hindex_data'.format(self.suffix_cuts):
            target_alias = 'h'
        elif self.target_table == \
                'analysis{0}_nc_data'.format(self.suffix_cuts):
            join_nc = True
            target_alias = 'nc'
        else:
            raise Exception("Not implemented")

        sql = """
            SELECT h.hindex_before, {0}.predict_after_years, {0}.{1}
            FROM analysis{2}_authors AS a""".format(
                target_alias, self.target_field, self.suffix_cuts)
        if join_nc:
            sql += """
                INNER JOIN analysis{0}_hindex_data AS h
                ON h.author_id = a.author_id AND
                h.predict_after_years = %(predict_after_years)s
                """.format(self.suffix_cuts)
            sql += """
                INNER JOIN analysis{0}_nc_data AS nc
                ON nc.author_id = a.author_id
                """.format(self.suffix_cuts)
        else:
            sql += """
                INNER JOIN analysis{0}_hindex_data AS h
                ON h.author_id = a.author_id
                """.format(self.suffix_cuts)

        sql += "WHERE a.train = 1 "

        if not all_years:
            sql += """
                AND {0}.predict_after_years = %(predict_after_years)s
                """.format(target_alias)

        sql += "ORDER BY a.author_id ASC"

        numauthors = c.execute(sql, {
            'predict_after_years': self.predict_after_years})

        hindex_train_data = np.fromiter(
                c, count=numauthors,
                dtype=[('hindex_before', 'i4'),
                       ('predict_after_years', 'i4'),
                       ('target', 'i4')])

        # Very simple model - average nc_after for each hindex_before
        c.execute("""
            SELECT MAX(hindex_before)
            FROM analysis{0}_hindex_data""".format(self.suffix_cuts))
        max_hindex = c.fetchone()[0]
        tmpsum = {}
        tmpcount = {}

        for (hindex_before, predict_after_years, target) in hindex_train_data:

            if predict_after_years not in tmpsum:
                tmpsum[predict_after_years] = \
                    {i: 0 for i in range(0, max_hindex+1)}
                tmpcount[predict_after_years] = \
                    {i: 0 for i in range(0, max_hindex+1)}

            tmpsum[predict_after_years][hindex_before] += \
                self.target_func(target)
            tmpcount[predict_after_years][hindex_before] += 1

        min_hindex_before_statistics = 10
        max_hindex_before = None

        predictor = {}
        for predict_after_years in tmpsum:
            predictor[predict_after_years] = {}
            for i in range(0, max_hindex+1):
                count = tmpcount[predict_after_years][i]
                sum = tmpsum[predict_after_years][i]
                if count > 0:
                    predictor[predict_after_years][i] = sum/count
                    # print(i, count, sum/count)
                else:
                    # print(i, count, "--")
                    pass

                # We will only evaluate (see evaluate) up to hindices which
                # are represented by at least 10 authors. Otherwise, our simple
                # averaging hindex_predictor is questionable due to lack of
                # statistics
                if max_hindex_before is None and \
                        count < min_hindex_before_statistics:
                    max_hindex_before = i-1

            # Fit linear function to get a value for empty spots
            x = list(predictor[predict_after_years].keys())
            y = [predictor[predict_after_years][k] for k in x]
            p = np.poly1d(np.polyfit(x, y, deg=1))
            for i in range(0, max_hindex+1):
                if i not in predictor[predict_after_years]:
                    predictor[predict_after_years][i] = p(i)
                    # print(i, "fit", p(i))

        # print("max_hindex_before", max_hindex_before)

        if all_years:
            return max_hindex_before, predictor
        else:
            return max_hindex_before, predictor[self.predict_after_years]

    def do_metrics(self, ytrue, ypred):
        from keras.losses import mean_squared_error, \
            mean_absolute_percentage_error
        import keras.backend as K
        from sklearn.metrics import r2_score

        r2 = r2_score(y_true=ytrue, y_pred=ypred)
        r = np.corrcoef(
                ytrue,
                ypred)[1, 0]

        loss = K.eval(mean_squared_error(
            K.variable(value=ytrue),
            K.variable(value=ypred)))

        # NOTE: We do have people with hindex_cumulative == 0 for which we
        # cannot calculate a "percentage error"!
        nonzero = ytrue > 0
        mape = K.eval(mean_absolute_percentage_error(
            y_true=K.variable(value=ytrue[nonzero]),
            y_pred=K.variable(value=ypred[nonzero])))

        return loss, r, r2, mape

    def linear_in_time_predictor(self, prediction_sql=None):

        authors = self.get_train_authors()
        authors = authors['author_id'][self.pick_validation_authors(authors)]
        authors = ','.join([str(id) for id in authors])

        c = db().cursor()

        if prediction_sql is None:
            if self.target_field == 'hindex_cumulative':
                prediction_sql = """t.hindex_before *
                    DATEDIFF(%(end_date)s, aa.first_paper_date) /
                    DATEDIFF(%(cutoff_date)s, aa.first_paper_date)"""
            elif self.target_field == 'nc_after':
                prediction_sql = """SQRT(t.nc_before * (
                    DATEDIFF(%(end_date)s, aa.first_paper_date)^2 /
                    DATEDIFF(%(cutoff_date)s, aa.first_paper_date)^2 - 1))"""
            else:
                raise Exception("Not implemented")


        result = {}
        for years in range(1, self.predict_after_years+1):
            end_date = self.cutoff_date + \
                relativedelta(years=years)

            sql = ("""
                SELECT
                    """+prediction_sql+""",
                    t."""+self.target_field+"""
                FROM analysis{0}_authors AS aa

                INNER JOIN """+self.target_table+""" AS t
                ON t.author_id = aa.author_id AND
                t.predict_after_years = %(years)s

                WHERE
                aa.author_id IN({1})
                """).format(self.suffix_cuts, authors)

            numauthors = c.execute(sql, {
                'end_date': end_date,
                'cutoff_date': self.cutoff_date,
                'years': years})

            data = np.fromiter(
                c, count=numauthors,
                dtype=[('ypred', 'f4'),
                       ('ytrue', 'f4')])
            ytrue = self.target_func(data['ytrue'])
            ypred = data['ypred']

            loss, r, r2, mape = self.do_metrics(ytrue=ytrue, ypred=ypred)

            print("Years", years)
            print("loss =", loss)
            print("r =", r)
            print("R^2 =", r2)
            print("MAPE =", mape)

            result[years] = [years, loss, r, r2, mape]

        return result

    def plusk_evaluate(self):
        if self.target_field == 'hindex_cumulative':
            prediction_sql = """t.hindex_before +
                DATEDIFF(%(end_date)s, %(cutoff_date)s) /
                DATEDIFF(CURDATE(), DATE_SUB(CURDATE(), INTERVAL 1 YEAR)) *
                .302"""
        elif self.target_field == 'nc_after':
            raise Exception("Not implemented (nc_after)")
        else:
            raise Exception("Not implemented")

        return self.linear_in_time_predictor(prediction_sql=prediction_sql)

    def hindex_predict(self, hindex_predictor=None):
        # Load hindex predictor to compare net to
        if hindex_predictor is None:
            print("Loading hindex data...")
            _, hindex_predictor = self.get_hindex_predictor()

        # Load author ids to validate on
        authors = self.get_train_authors()
        authors = authors['author_id'][self.pick_validation_authors(authors)]
        authors = ','.join([str(id) for id in authors])

        c = db().cursor()
        numauthors = c.execute("""
            SELECT h.hindex_before
            FROM analysis{0}_authors AS a
            INNER JOIN analysis{0}_hindex_data AS h
            ON h.author_id = a.author_id AND
            h.predict_after_years = %(predict_after_years)s
            WHERE
            a.author_id IN({1})
            ORDER BY a.author_id ASC
            """.format(self.suffix_cuts, authors), {
                'predict_after_years': self.predict_after_years})

        y_hindex = np.array([hindex_predictor[row[0]] for row in c])

        return y_hindex

    def hindex_evaluate(self, y=None, hindex_predictor=None):
        # Load y
        if y is None:
            y = self.load_validation_data_y()
        y = y.reshape(-1)

        # naive h-index prediction
        y_hindex = self.hindex_predict(hindex_predictor=hindex_predictor)
        # Consistency check
        if len(y_hindex) != len(y):
            raise Exception("numauthors don't match?!")

        return self.do_metrics(ytrue=y, ypred=y_hindex)

    def do_evaluate(self, y, y_net):
        loss, r, r2, mape = self.do_metrics(ytrue=y, ypred=y_net)

        print(self.target)
        print("-> Loss", loss)
        print("-> R^2_net", r2)
        print("-> r_net", r)
        print("-> MAPE_net", mape)

        return loss, r, r2, mape

    def evaluate(self):

        # Load data
        print("Loading net evaluation data...")
        inputs, y = self.load_validation_inputs()

        # Speed up batch evaluations
        self.clear_keras_session()

        model = self.load_model()
        y_net = model.predict(inputs)

        return self.do_evaluate(y, y_net)

    def plot_correlation(self):
        
        # Load data
        print("Loading net evaluation data...")
        inputs, y = self.load_validation_inputs()
        
        model = self.load_model()
        
        # Calculate correlation coefficients
        y_net = model.predict(inputs)
        
        if isinstance(self, TimeSeriesNet):
            
            if self.force_monotonic:
                y = np.cumsum(y, axis=1)
                y_net = np.cumsum(y_net, axis=1)
            
            y = y[:, self.predict_after_years-1]
            y_net = y_net[:, self.predict_after_years-1]
        
        print(np.corrcoef(y, y_net))
        
        filename = os.path.join(self.evaluate_dir,
                                'correlation-%s.csv' % self.suffix)
        pd.DataFrame(data={'y': y, 'y_net': y_net}).to_csv(filename)

    def plot_correlation_hindex(self):

        # Load data
        print("Loading net evaluation data...")
        y = self.load_validation_data_y()

        # Calculate correlation coefficients
        if isinstance(self, TimeSeriesNet):
            if self.force_monotonic:
                y = np.cumsum(y, axis=1)
            y = y[:, self.predict_after_years-1]

        y_hindex = self.hindex_predict()

        print(np.corrcoef(y, y_hindex))

        filename = os.path.join(self.evaluate_dir,
                                'correlation-hindex-%s.csv' % self.suffix)
        pd.DataFrame(data={'y': y, 'y_hindex': y_hindex}).to_csv(filename)

    def plot_example_trajectories(self):
        
        if not self.target.startswith('hindex_cumulative'):
            raise Exception("Only implemented for hindex_cumulative")
        
        authors = self.get_train_authors()
        validation_authors = \
            authors['author_id'][self.pick_validation_authors(authors)]
        validation_authors = ','.join([str(id) for id in validation_authors])
        
        # Randomly select authors
        author_ids = []
        selectrules = [
            (10, 2, 7),
            (10, 7, 15),
            (10, 15, 30),
            (10, 30, 40)
            ]
        c = db().cursor()
        for (num, min_hindex, max_hindex) in selectrules:
            sql = """
                SELECT a.author_id
                FROM analysis{0}_authors AS a
                INNER JOIN analysis{0}_hindex_data AS h
                ON h.author_id = a.author_id AND
                    h.predict_after_years = %(predict_after_years)s
                WHERE
                a.author_id IN ({1}) AND
                h.hindex_cumulative < %(max_hindex)s AND
                h.hindex_cumulative > %(min_hindex)s
                ORDER BY RAND()
                LIMIT %(num)s""".format(self.suffix_cuts, validation_authors)
            c.execute(sql, {
                'predict_after_years': self.predict_after_years,
                'max_hindex': max_hindex,
                'min_hindex': min_hindex,
                'num': num})
            author_ids += [row[0] for row in c]
        
        # Order by author_id ASC to match self.load_*_data
        author_ids.sort()
        
        indices = np.where(np.isin(authors['author_id'], author_ids))

        result = {}
        columns = []
        
        # Load x data
        inputs, y = self.load_validation_inputs(indices=indices)
        if isinstance(self, TimeSeriesNet):
            if self.force_monotonic:
                y = np.cumsum(y, axis=1)
        
        # Generate y data for neural net
        orig_predict_after_years = self.predict_after_years
        orig_suffix = self.suffix
        
        if isinstance(self, TimeSeriesNet):
            years_range = [self.predict_after_years]
        else:
            years_range = range(1, self.predict_after_years+1)
        for years in years_range:
            print("Years:", years)
            # Set predict_after_years
            self.predict_after_years = years
            
            # For file names
            if not isinstance(self, TimeSeriesNet):
                self.suffix = orig_suffix + '-years' + str(years)
            
            # Load keras model
            model = self.load_model()
            y_net = model.predict(inputs)
            
            if isinstance(self, TimeSeriesNet):
                
                if self.force_monotonic:
                    y_net = np.cumsum(y_net, axis=1)
                
                for years2 in range(1, orig_predict_after_years+1):
                    result[years2] = []
                    for i, author_id in enumerate(author_ids):
                        if years2 == orig_predict_after_years:
                            columns.append([str(author_id) + " true",
                                            str(author_id) + " net"])
                        result[years2].append([y[i, years2-1],
                                               y_net[i, years2-1]])
            else:
                y = self.load_validation_data_y()
                result[years] = []
                for i, author_id in enumerate(author_ids):
                    if years == orig_predict_after_years:
                        columns.append([str(author_id) + " true",
                                        str(author_id) + " net"])
                    result[years].append([y[i], y_net[i]])
 
        self.predict_after_years = orig_predict_after_years
        self.suffix = orig_suffix
        
        # Sort result by hindex for easier stuff
        indexes = list(range(len(result[self.predict_after_years])))
        indexes.sort(key=lambda x: result[self.predict_after_years][x][0],
                     reverse=True)
        for years in result:
            result[years] = list(map(result[years].__getitem__, indexes))
        columns = list(map(columns.__getitem__, indexes))
        
        # Flatten and add years
        columns = ['years'] + [e for tmp in columns for e in tmp]
        data = []
        for years, years_data in result.items():
            data.append([years] + [e for tmp in years_data for e in tmp])
                       
        # Write result to csvfile
        filename = os.path.join(self.evaluate_dir,
                                'example-trajectories-%s.csv' % self.suffix)
        pd.DataFrame(data=data, columns=columns).to_csv(filename)


class TimeSeriesNet(Net):
    
    def __init__(self, cutoff, target, force_monotonic,
                 first_paper_range=None):
        
        self.force_monotonic = force_monotonic
        
        super().__init__(cutoff, target, first_paper_range)
        
        # Important since y files get saved with this suffix!
        self.target += '_time'
        if self.force_monotonic:
            self.target += '_mono'
            self.suffix += '_mono'
    
    def get_net_filename(self):
        return os.path.join(self.data_dir, 'models',
                            'net_time-%s.h5' % self.suffix)
    
    def generate_data_y(self, train=None, max_hindex_before=None,
                        author_generator=None):
        
        print("Generating y data...")
        
        # Get author generator
        if author_generator:
            numauthors, author_generator = author_generator
        else:
            numauthors, author_generator = self. \
                get_data_author_generator(max_hindex_before, train=train)
      
        # Output data
        y = np.zeros((numauthors, self.predict_after_years))
        
        c = db().cursor()
        
        for i, row in author_generator():
            author_id = row[0]
            first_paper_date = row[1]
            
            print(author_id, first_paper_date)
            
            sql = """
                SELECT t.%s
                FROM %s AS t
                """ % (self.target_field, self.target_table)
            sql += """
                WHERE t.author_id = %(author_id)s
                ORDER BY t.predict_after_years ASC
                """
            
            numyears = c.execute(sql, {'author_id': author_id})
            if numyears != self.predict_after_years:
                raise Exception("numyears != self.predict_after_years - run "
                                "generate_*_data()?")
            
            if self.force_monotonic:
                prev = 0
                for j, (raw_target,) in enumerate(c.fetchall()):
                    target = self.target_func(raw_target)
                    delta = target - prev
                    y[i][j] = delta
                    prev = target
            else:
                y[i] = self.target_func(np.fromiter(c,
                                                    dtype=[('target', 'f4')],
                                                    count=numyears)['target'])
        
        columns = ['y' + str(i) for i in range(1, self.predict_after_years+1)]
        return pd.DataFrame(y, columns=columns)
     
    def train(self, activation='tanh', live_validate=False, load=False):
        
        # Load validation data
        if live_validate:
            validation_data = self.load_validation_inputs()
        else:
            validation_data = None

        # Speed up batch evaluations
        self.clear_keras_session()

        # Load data
        print("Loading net training data...")
        x, xaux, y = self.load_train_data()

        # Build/load keras model
        if load:
            model = self.load_model()
        else:
            print("Loading keras / build model...")
            from keras.models import Model
            from keras.layers import Input, Dense, Conv1D, concatenate, \
                GlobalAveragePooling1D
            
            perpaper_inputs = Input(shape=x[0].shape,
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
            lossfunction = 'mean_squared_error'
            
            model = Model(inputs=[perpaper_inputs, perauthor_inputs],
                          outputs=outputs)
            
            model.compile(
                loss=lossfunction,
                optimizer='adam')
        
        # Fit keras model
        model.fit({'perpaper_inputs': x, 'perauthor_inputs': xaux}, y,
                  epochs=self.epochs, batch_size=50,
                  validation_data=validation_data)
        model.save(self.get_net_filename())
    
    def hindex_evaluate(self, y=None, hindex_predictor=None):
        
        # h-index predictor
        if hindex_predictor is None:
            print("Loading h-index data...")
            _, hindex_predictor = self.get_hindex_predictor(all_years=True)
        
        # Load ytrue
        if y is None:
            y = self.load_validation_data_y()
            # For monotic function need to calculate target from the deltas
            if self.force_monotonic:
                y = np.cumsum(y, axis=1)
                
        result = {}
        
        orig_predict_after_years = self.predict_after_years
        for years in range(1, self.predict_after_years+1):
            
            print("Years", years)
            
            self.predict_after_years = years
            loss, r, r2, mape = super().hindex_evaluate(
                y[:, years-1],
                hindex_predictor=hindex_predictor[years])
            self.predict_after_years = orig_predict_after_years
           
            print("-> r_hindex", r)
            print("-> R^2 hindex", r2)
            print("-> MAPE hindex", mape)
            
            result[years] = [years, loss, r, r2, mape]
    
        return result

    def do_evaluate(self, y, y_net):

        # For monotic function need to calculate target from the deltas
        if self.force_monotonic:
            y_net = np.cumsum(y_net, axis=1)
            y = np.cumsum(y, axis=1)

        # Evaluate metrics on a yearly basis
        result = {}
        from keras.losses import mean_squared_error
        import keras.backend as K
        from sklearn.metrics import r2_score
        for years in range(1, self.predict_after_years+1):

            index = years - 1
            ytrue = y[:, index]
            ypred = y_net[:, index]

            loss, r, r2, mape = self.do_metrics(ytrue=ytrue, ypred=ypred)

            print("Years ", years)
            print("-> Loss", loss)
            print("-> R^2_net", r2)
            print("-> r_net", r)
            print("-> MAPE_net", mape)

            result[years] = [years, loss, r, r2, mape]

        return result


if __name__ == '__main__':
    """
    a = Net(cutoff=Net.CUTOFF_SINGLE, target='hindex_cumulative')
    a.choose_train_authors()
    """
    
    """
    n = Net(cutoff=Net.CUTOFF_SINGLE, target='hindex_cumulative')
    # n.train(live_validate=True)
    # Continue training previously trained net
    # n.train(live_validate=True, load=False)
    # n.evaluate()
    # n.test()
    """
    
    """
    n = Net(cutoff=Net.CUTOFF_SINGLE, target='sqrt_nc_after')
    # n = Net(cutoff=Net.CUTOFF_SINGLE, target='hindex_cumulative')
    n.suffix += '-test-1year'
    n.predict_after_years = 1
    # n.epochs = 5
    # n.epochs = 40
    # n.train(live_validate=True, load=True)
    n.train(live_validate=True)
    n.evaluate()
    """
    
    """
    # n = TimeSeriesNet(cutoff=Net.CUTOFF_SINGLE, target='hindex_cumulative',
    #                   force_monotonic=False)
    n = TimeSeriesNet(cutoff=Net.CUTOFF_SINGLE, target='hindex_cumulative',
                      force_monotonic=True)
    # n = TimeSeriesNet(cutoff=Net.CUTOFF_SINGLE, target='sqrt_nc_after',
    #                  force_monotonic=True)
    # n.train(live_validate=True)
    n.evaluate()
    """
    
    """
    n = Net(cutoff=Net.CUTOFF_SINGLE, target='sqrt_nc_after')
    # n = TimeSeriesNet(cutoff=Net.CUTOFF_SINGLE, target='hindex_cumulative',
    #                   force_monotonic=True)
    n.suffix += '-test-no-topics-no-categories'
    n.set_exclude_data(['categories',
                        'paper_topics'])
    n.evaluate()
    """

    n = TimeSeriesNet(cutoff=Net.CUTOFF_SINGLE, target='hindex_cumulative',
                      force_monotonic=True)
    n.choose_train_authors()
