import os
import numpy as np
import pandas as pd
from shutil import copy
from net import Net, TimeSeriesNet


class Task:

    def __init__(self, net, years_loop=True, epochs_step=5, epochs_steps=2,
                 subsets_evaluations=[]):
        self.net = net
        self.years_loop = years_loop

        self.epochs_step = epochs_step
        self.epochs_steps = epochs_steps

        self.subsets_evaluations = subsets_evaluations

        # Create dir for net files if it doesn't exist
        self.should_save = False

        self.data_dir = os.path.join(self.net.evaluate_dir, 'task-results')

        os.makedirs(self.data_dir, exist_ok=True)

    def __enter__(self):

        self.results = {}

        self.orig_suffix = self.net.suffix
        self.orig_epochs = self.net.epochs
        self.orig_predict_after_years = self.net.predict_after_years

        return self

    def get_results_filename(self):
        return os.path.join(
            self.data_dir,
            self.__class__.__name__ + self.orig_suffix + '.npy')

    def save_results(self):
        np.save(self.get_results_filename(), self.results)
        self.save_results_nice()

    def load_results(self):
        return np.load(self.get_results_filename()).item()

    def save_results_nice(self):
        pass

    def __exit__(self, *args, **kwargs):

        # Write results
        if self.should_save:
            self.save_results()

        self.net.suffix = self.orig_suffix
        self.net.epochs = self.orig_epochs
        self.net.predict_after_years = self.orig_predict_after_years

    def get_years_range(self):
        if isinstance(self.net, TimeSeriesNet) or not self.years_loop:
            return [self.net.predict_after_years]
        else:
            return range(1, self.net.predict_after_years+1)

    def get_suffix(self, additional_epochs, **kwargs):
        suffix = self.orig_suffix
        if not isinstance(self.net, TimeSeriesNet) and self.years_loop:
            suffix += '-years' + str(self.net.predict_after_years)
        if additional_epochs > 0:
            suffix += '-add'+str(additional_epochs)
        return suffix

    def generator(self):

        for years in self.get_years_range():

            print("Years:", years)

            # Set predict_after_years
            self.net.predict_after_years = years

            for i in range(0, self.epochs_steps+1):

                if i > 0:
                    # Continue training for a few epochs to get to variance of
                    # individual epochs
                    self.net.epochs = self.epochs_step
                else:
                    # Reset from additional epochs training
                    self.net.epochs = self.epochs = self.orig_epochs

                yield (i*self.epochs_step,)

    def train(self, params):
        additional_epochs, *args = params
        if additional_epochs > 0:
            load = True
            args = tuple(args)

            self.net.suffix = self.get_suffix(
                additional_epochs - self.epochs_step, *args)
            prev_net_file = self.net.get_net_filename()

            self.net.suffix = self.get_suffix(additional_epochs, *args)
            net_file = self.net.get_net_filename()

            copy(prev_net_file, net_file)
        else:
            load = False

        self.net.train(live_validate=False, load=load)

    def update_evaluate_result(self, params, data):
        additional_epochs, *_ = params
        if additional_epochs not in self.results:
            self.results[additional_epochs] = []
        self.results[additional_epochs].append(data)

    def evaluate(self, params):

        net_result = self.net.evaluate()

        if isinstance(self.net, TimeSeriesNet):
            for years, row in net_result.items():
                if self.years_loop or years == self.net.predict_after_years:
                    self.update_evaluate_result(params, row)
        else:
            loss, r, r2, mae = net_result
            years = self.net.predict_after_years
            self.update_evaluate_result(params, [years, loss, r, r2, mae])

    def train_all(self):
        for params in self.generator():
            self.train(params)

    def evaluate_all(self):
        self.results = {}
        self.should_save = True
        for params in self.generator():
            self.evaluate(params)

    def run(self):
        for params in self.generator():
            self.train(params)
            self.evaluate(params)

    def subsets_evaluate(self):
        for params in self.generator():
            # Only additional epochs 0
            additional_epochs, *_ = params
            if additional_epochs != 0:
                continue

            for subsetsclass, runs in self.subsets_evaluations:
                subsets_evaluate = subsetsclass(self.net, runs=runs)
                subsets_evaluate.run()


class PredictivityOverTime(Task):

    def save_results_nice(self):
        # Write results
        columns = ['years', 'loss', 'r', 'R^2', 'MAPE']
        for additional_epochs, data in self.results.items():

            suffix = self.orig_suffix
            if additional_epochs > 0:
                suffix += '-add'+str(additional_epochs)

            filename = os.path.join(
                self.data_dir,
                'predictivity-over-time-' + suffix + '.csv')
            pd.DataFrame(data, columns=columns).to_csv(filename)

    def generator(self):

        for params in super().generator():

            # For file names
            self.net.suffix = self.get_suffix(*params)

            yield params


class PredictivityOverTimeNoHindex0(PredictivityOverTime):

    def get_nonzero_author_ids(self):
        filename = os.path.join(self.net.data_dir, 'author_ids-%s.npy' %
                                self.orig_suffix)
        try:
            nonzero_author_ids = np.load(filename)
        except FileNotFoundError as e:
            from db import db
            sql = """
                SELECT a.author_id
                FROM analysis{0}_authors AS a

                INNER JOIN analysis{0}_hindex_data AS h
                ON h.author_id = a.author_id

                WHERE
                h.predict_after_years = 1 AND
                h.hindex_cumulative = 0
                """.format(self.net.suffix_cuts)
            c = db().cursor()
            numauthors = c.execute(sql)
            nonzero_author_ids = np.fromiter(
                c, count=numauthors, dtype=[('author_id', 'i4')])['author_id']

            np.save(filename, nonzero_author_ids)

        return nonzero_author_ids

    def __enter__(self):
        ret = super().__enter__()


        # Change suffix
        self.__orig_suffix_no_hindex0 = self.orig_suffix
        self.orig_suffix += '-nohindex0'
        self.net.suffix = self.orig_suffix

        # Change get_train_authors method
        self.orig_get_train_authors = self.net.__class__.get_train_authors
        def fake_get_train_authors(net, *args, **kwargs):
            authors = self.orig_get_train_authors(net, *args, **kwargs)

            print('authors train incl 0 hindex',
                  len(np.where(authors['train'] == 1)[0]))

            nonzero_author_ids = self.get_nonzero_author_ids()
            indices = np.where(np.isin(authors['author_id'], nonzero_author_ids))
            authors['train'][indices] = 0

            print('authors train excl 0 hindex',
                  len(np.where(authors['train'] == 1)[0]))

            return authors

        self.net.__class__.get_train_authors = fake_get_train_authors

        return ret

    def __exit__(self, *args, **kwargs):
        super().__exit__(*args, **kwargs)

        # Reset suffix
        self.net.suffix = self.__orig_suffix_no_hindex0

        # Restore get_train_authors method
        self.net.__class__.get_train_authors = self.orig_get_train_authors


class DataImportance(Task):

    allow_includes = False

    default_excludes = [
        ['broadness_lda'],
        ['num_citations'],
        ['months'],
        ['pagerank'],
        ['paper_topics'],
        ['length'],
        ['jif',
         'published'],
        ['num_coauthors'],
        ['avg_coauthor_pagerank',
         'max_coauthor_pagerank',
         'min_coauthor_pagerank'],
        ['categories']
        ]

    def __init__(self, net, excludes=None, includes=None, **kwargs):
        super().__init__(net, years_loop=False, **kwargs)
        if includes is not None and  excludes is not None:
            raise Exception("Can only specify one of excludes and includes")
        if not self.allow_includes and includes is not None:
            raise Exception("Please use dedicated classes for includes")

        if includes:
            self.includes = includes
            self.excludes = [self.__invert(include) for include in includes]
        elif excludes:
            self.includes = None
            self.excludes = excludes
        else:
            self.includes = None
            self.excludes = self.default_excludes

    def __invert(self, fields):
        # 'padding' should not be inverted!
        return [field for field in sorted(Net.data_positions.keys())
                if field not in fields and field != 'padding'] + \
               [field for field in sorted(Net.data_positions_aux.keys())
                if field not in fields]

    def __enter__(self):
        ret = super().__enter__()

        # Put include in suffix
        if self.includes is not None:
            self.__orig_suffix_no_include = self.orig_suffix
            self.orig_suffix += '-include'
            self.net.suffix = self.orig_suffix

        return ret

    def __exit__(self, *args, **kwargs):
        super().__exit__(*args, **kwargs)

        # Reset exclude
        self.net.set_exclude_data([])

        # Remove include in suffix
        if self.includes is not None:
            self.net.suffix = self.__orig_suffix_no_include

    def save_results_nice(self):
        columns = ['exclude', 'years', 'loss', 'r', 'R^2', 'MAPE']
        for additional_epochs, data in self.results.items():

            suffix = self.orig_suffix
            if additional_epochs > 0:
                suffix += '-add'+str(additional_epochs)

            filename = os.path.join(self.data_dir,
                                    'data-importance-' + suffix + '.csv')
            pd.DataFrame(data, columns=columns).to_csv(filename)

    def get_suffix(self, additional_epochs, exclude):
        suffix = super().get_suffix(additional_epochs)
        if self.includes is not None:
            suffix += '-' + '-'.join(self.__invert(exclude))
        else:
            suffix += '-' + '-'.join(exclude)
        return suffix

    def generator(self):
        for exclude in self.excludes:
            for params in super().generator():

                params = params + (exclude,)

                # For file names
                self.net.suffix = self.get_suffix(*params)

                # Set exclude
                self.net.set_exclude_data(exclude)

                yield params

    def update_evaluate_result(self, params, data):
        additional_epochs, exclude = params
        if additional_epochs not in self.results:
            self.results[additional_epochs] = []
        self.results[additional_epochs].append(['-'.join(exclude)] + data)


class DataImportanceInclude(DataImportance):
    allow_includes = True


class DataImportanceOverTime(DataImportance):

    def __init__(self, net, **kwargs):
        super().__init__(net, **kwargs)
        self.years_loop = True

    def save_results_nice(self):

        if not len(self.results):
            return

        base_columns = ['loss', 'r', 'R^2', 'MAPE']
        if self.includes is not None:
            columns = ["%s_%s" % (col, include[0])
                       for include in self.includes for col in base_columns]
        else:
            columns = ["%s_%s" % (col, exclude[0])
                    for exclude in self.excludes for col in base_columns]
        columns = ['years'] + columns

        for i in range(0, self.epochs_steps + 1):
            additional_epochs = i * self.epochs_step

            data = []
            for years, year_results in self.results[additional_epochs].items():
                row = [year_results['-'.join(exclude)][i]
                       for exclude in self.excludes
                       for i, _ in enumerate(base_columns)]
                data.append([years] + row)

            # Save data
            suffix = self.orig_suffix
            if additional_epochs > 0:
                suffix += '-add' + str(additional_epochs)

            filename = os.path.join(
                self.data_dir,
                'data-importance-over-time-' + suffix + '.csv')
            pd.DataFrame(data, columns=columns).to_csv(filename)

    def get_years_range(self):
        if isinstance(self.net, TimeSeriesNet):
            return super().get_years_range()
        else:
            # To save time only 1, 5, 10
            return [1, 5, 10]

    def update_evaluate_result(self, params, data):
        additional_epochs, exclude = params
        exclude = '-'.join(exclude)
        years = data[0]
        data = data[1:]

        if additional_epochs not in self.results:
            self.results[additional_epochs] = {}
        if years not in self.results[additional_epochs]:
            self.results[additional_epochs][years] = {}
        self.results[additional_epochs][years][exclude] = data


class DataImportanceOverTimeInclude(DataImportanceOverTime):
    allow_includes = True


if __name__ == '__main__':
    """
    net = Net(cutoff=Net.CUTOFF_SINGLE, target='hindex_cumulative')
    net.epochs = 1
    net.predict_after_years = 1
    with PredictivityOverTime(net) as task:
        task.train_all()
        task.evaluate_all()
    """

    """
    net = Net(cutoff=Net.CUTOFF_SINGLE, target='sqrt_nc_after')
    net.epochs = 1

    excludes = DataImportance.default_excludes + [
        [],
        ['broadness_lda',
         'months',
         'pagerank',
         'length',
         'jif',
         'published',
         'num_coauthors',
         'avg_coauthor_pagerank',
         'max_coauthor_pagerank',
         'min_coauthor_pagerank',
         'categories',
         'paper_topics']
        ]
    with DataImportance(net, excludes) as task:
        task.epochs_steps = 0
        task.train_all()
        task.evaluate_all()
    """

    """
    net = Net(cutoff=Net.CUTOFF_SINGLE, target='hindex_cumulative')
    net.epochs = 1
    with DataImportanceOverTime(net) as task:
        # task.train_all()
        task.evaluate_all()
    """

    """
    net = TimeSeriesNet(cutoff=Net.CUTOFF_SINGLE, target='hindex_cumulative',
                        force_monotonic=True)
    net.epochs = 1
    with DataImportanceOverTime(net, epochs_steps=0) as task:
        # task.train_all()
        task.evaluate_all()
    """
