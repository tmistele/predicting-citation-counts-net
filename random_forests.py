import numpy as np
from net import TimeSeriesNet, Net
import os


class RandomForestTimeSeries(TimeSeriesNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.suffix += '-rf'

    def get_net_filename(self):
        return os.path.join(self.data_dir, 'models',
                            'rf_time-%s.pkl' % self.suffix)

    def __net_to_rf_inputs(self, data):
        x, xaux, y = data
        # Reshape from (nsamples, npapers, nfeatures) to
        # (nsamples, npapers*nfeatures)
        x = np.reshape(x, (x.shape[0], x.shape[1]*x.shape[2]))
        # Add xaux data to x
        x = np.concatenate((x, xaux), axis=1)
        return x, y

    def load_validation_inputs(self, *args, **kwargs):
        return self.__net_to_rf_inputs(
            super().load_validation_data(*args, **kwargs))

    def load_model(self):
        print("Loading model...")
        from sklearn.externals import joblib
        return joblib.load(self.get_net_filename())

    def train(self, load=False, live_validate=False):

        if load:
            raise Exception("retraining not supported with random forests")
        if live_validate:
            raise Exception("live_validate not supported with random forests")

        # Load data
        print("Loading net training data...")
        x, y = self.__net_to_rf_inputs(self.load_train_data())

        # As in https://github.com/Lucaweihs/impact-prediction/blob/master/models.py#L129
        from multiprocessing import cpu_count
        rf_params = {
            'n_estimators': 1500,
            'max_features': .3333,
            'min_samples_leaf': 25,
            'n_jobs': cpu_count() - 1,
            'verbose': 1}

        from sklearn.ensemble import RandomForestRegressor
        regr = RandomForestRegressor(**rf_params)

        # Fit and save model
        print("Fitting...")
        regr.fit(x, y)

        from sklearn.externals import joblib
        joblib.dump(regr, self.get_net_filename())


if __name__ == '__main__':

    rf = RandomForestTimeSeries(cutoff=Net.CUTOFF_SINGLE,
                                target='hindex_cumulative',
                                force_monotonic=True)

    from runner import CrossValidation
    with CrossValidation(rf) as cv:
        for _ in cv(load=[0], load_to_db=False):

            """
            exclude = [
                'broadness_lda',
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
            """

            """
            exclude = [
                'broadness_lda',
                'categories',
                'paper_topics']
            rf.set_exclude_data(exclude)
            rf.suffix += '-' + '-'.join(exclude)
            """

            rf.train()
            rf.evaluate()

    net = TimeSeriesNet(cutoff=Net.CUTOFF_SINGLE,
                        target='hindex_cumulative',
                        force_monotonic=True)
    with CrossValidation(net) as cv:
        for _ in cv(load=[0], load_to_db=False):
            net.evaluate()
