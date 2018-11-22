import os
import numpy as np
import pandas as pd
from net import Net, TimeSeriesNet
from random_forests import RandomForestTimeSeries
from tasks import DataImportance, DataImportanceOverTime, \
    PredictivityOverTime, DataImportanceInclude, \
    DataImportanceOverTimeInclude, PredictivityOverTimeNoHindex0
from cross_validation import CrossValidation
from subsets_evaluate import SqrtNcAfterSubsetsEvaluation

class Runner:
     
    def __init__(self, net_prepare_func=None):
        self.num_cv = 20
        self.Net = Net
        self.net_params = {}
        self.net_prepare_func = net_prepare_func
        
        self.targets = {
            'hindex_cumulative': self.hindex_tasks,
            'sqrt_nc_after': self.sqrt_nc_tasks
            }

        self.rf_targets = {
            'hindex_cumulative': self.rf_hindex_tasks,
            'sqrt_nc_after': self.rf_sqrt_nc_tasks
            }

    def __prepare_net(self, net):
        # Mainly used for setting net.data_dir_xy on cluster and net.epochs
        # for debugging
        if self.net_prepare_func is not None:
            self.net_prepare_func(net)

        # Save results for ignore_max_hindex_before in different directory
        if net.ignore_max_hindex_before:
            net.evaluate_dir = os.path.join(net.evaluate_dir, 'no-max-hindex')
            os.makedirs(net.evaluate_dir, exist_ok=True)

        return net

    def create_rf(self, target):
        # Initialize random forests like a neural network
        orig_net = self.Net
        self.Net = RandomForestTimeSeries
        rf = self.Net(cutoff=self.Net.CUTOFF_SINGLE, target=target,
                      **self.net_params)
        self.Net = orig_net
        return self.__prepare_net(rf)
    
    def create_net(self, target):
        net = self.Net(cutoff=self.Net.CUTOFF_SINGLE, target=target,
                       **self.net_params)
        return self.__prepare_net(net)
    
    def hindex_tasks(self, net):
        with PredictivityOverTime(net, epochs_steps=2) as task:
            yield task
        with DataImportanceOverTime(net, epochs_steps=0) as task:
            yield task

        includes = DataImportanceOverTime.default_excludes
        with DataImportanceOverTimeInclude(net, includes=includes,
                                           epochs_steps=0) as task:
            yield task

        with PredictivityOverTimeNoHindex0(net, epochs_steps=0) as task:
            yield task

    def sqrt_nc_tasks(self, net):
        subsets_evaluations = [
            (SqrtNcAfterSubsetsEvaluation, None)
        ]
        excludes = DataImportance.default_excludes + [ [] ]
        with DataImportance(net, excludes=excludes, epochs_steps=0,
                            subsets_evaluations=subsets_evaluations) as task:
            yield task

        includes = DataImportance.default_excludes
        with DataImportanceInclude(net, includes=includes,
                                   epochs_steps=0) as task:
            yield task

    def rf_hindex_tasks(self, rf):
        with PredictivityOverTime(rf, epochs_steps=0) as task:
            yield task

    def rf_sqrt_nc_tasks(self, rf):
        with DataImportance(rf, excludes=[ [] ], epochs_steps=0) as task:
            yield task
        with DataImportanceInclude(rf, includes=[ ['num_citations'] ],
                                   epochs_steps=0) as task:
            yield task

    def prepare(self):
        # Generate author ids and make sure everything is prepared for training
        # without db
        
        # Make sure author_id are generated (shared by both nets)
        net = self.create_net('hindex_cumulative')
        with CrossValidation(net) as cv:
            for _ in cv(num=self.num_cv, load=False):
                pass
        
        # Make sure x data is generated (shared by both nets)
        net.load_data_x()
        
        # Make sure y data are generated
        for target, tasks in self.targets.items():
            net = self.create_net(target)
        
            for task in tasks(net):
                for years in task.get_years_range():
                    net.predict_after_years = years
                    net.load_data_y()

        # Make sure author id files from NoHindex0 tasks are generated
        # May be nicer to put a general prepare() method in Task class
        for target, tasks in self.targets.items():
            net = self.create_net(target)
            with CrossValidation(net) as cv:
                for _ in cv(num=self.num_cv, load=True, load_to_db=True):
                    for task in tasks(net):
                        if isinstance(task, PredictivityOverTimeNoHindex0):
                            task.get_nonzero_author_ids()

    def train_net(self, i=None):
        # Train everything. No database here
        
        if i is None:
            load = True
        elif type(i) is int:
            load = i
        else:
            raise Exception("Invalid i")
         
        for target, tasks in self.targets.items():
            net = self.create_net(target)
        
            with CrossValidation(net) as cv:
                for _ in cv(num=self.num_cv, load=load, load_to_db=False):
                    for task in tasks(net):
                        task.train_all()

    def train_rf(self, i=None):
        # Train random forests like a neural network
        orig_targets = self.targets
        orig_create_net = self.create_net

        self.targets = self.rf_targets
        self.create_net = self.create_rf

        self.train_net(i=i)

        self.targets = orig_targets
        self.create_net = orig_create_net

    def train_one(self, i):
        # Train i
        self.train_net(i=i)
        self.train_rf(i=i)
    
    def evaluate_net(self, i=None):
        if i is None:
            load = True
        elif type(i) is int:
            load = i
        else:
            raise Exception("Invalid i")
        
        # Evaluate net performance
        for target, tasks in self.targets.items():
            net = self.create_net(target)
            
            with CrossValidation(net) as cv:
                for _ in cv(num=self.num_cv, load=load, load_to_db=False):
                    for task in tasks(net):
                        task.evaluate_all()

    def evaluate_rf(self, i=None):
        # Evaluate random forests like a neural network
        orig_targets = self.targets
        orig_create_net = self.create_net

        self.targets = self.rf_targets
        self.create_net = self.create_rf

        self.evaluate_net(i)

        self.targets = orig_targets
        self.create_net = orig_create_net
    
    def evaluate_linear_naive(self, i=None):
        if i is None:
            load = True
        elif type(i) is int:
            load = i
        else:
            raise Exception("Invalid i")
        
        # Evaluate naive hindex / linear in time predictor performance
        for target, tasks in self.targets.items():
            net = self.create_net(target)
            
            with CrossValidation(net) as cv:
                for _ in cv(num=self.num_cv, load=load, load_to_db=True):
                   
                    # Trick to use Task logic for naive hindex/linear in time
                    # predictors for both Net and TimeSeriesNet
                    orig_evaluate = net.evaluate
                    orig_suffix = net.suffix
                   
                    # naive h-index predictor
                    net.evaluate = net.hindex_evaluate
                    net.suffix = orig_suffix + '-naive'
                    with PredictivityOverTime(net, epochs_steps=0) as task:
                        task.evaluate_all()
                        
                    # linear in time predictor
                    net.evaluate = net.linear_in_time_predictor
                    net.suffix = orig_suffix + '-linear'
                    with PredictivityOverTime(net, epochs_steps=0) as task:
                        task.evaluate_all()
                    
                    # Revert trick
                    net.evaluate = orig_evaluate
                    net.suffix = orig_suffix
    
    def evaluate_one(self, i):
        self.evaluate_net(i)
        self.evaluate_linear_naive(i)
        self.evaluate_rf(i)
    
    def average_deep(self, data, mean=None, std=None):
        ref = data[0]
        if mean is None:
            from copy import deepcopy
            mean = deepcopy(ref)
        if std is None:
            from copy import deepcopy
            std = deepcopy(ref)

        if type(ref) is list:
            for i in range(0, len(ref)):
                # Allow 1D and 2D lists in final level
                # np.mean(, axis=0) could handle this, but we need to filter
                # out the non-numeric fields
                if type(ref[i]) is list:
                    for j in range(0, len(ref[i])):
                        if type(ref[i][j]) is str:
                            continue
                        values = [e[i][j] for e in data]
                        mean[i][j] = np.mean(values, axis=0)
                        std[i][j] = np.std(values, axis=0)
                else:
                    if type(ref[i]) is str:
                        continue
                    values = [e[i] for e in data]
                    mean[i] = np.mean(values, axis=0)
                    std[i] = np.std(values, axis=0)
            return
        
        for key in ref:
            self.average_deep([e[key] for e in data], mean[key], std[key])

        return mean, std
    
    def average_and_save(self, task, results):
        # Calculate mean / std
        mean, std = self.average_deep(results)
        
        orig_orig_suffix = task.orig_suffix
        orig_results = task.results
        orig_data_dir = task.data_dir
        
        # Save to ordinary net directory
        task.data_dir = task.net.evaluate_dir
        
        # Save mean
        task.orig_suffix = orig_orig_suffix + '-mean'
        task.results = mean
        task.save_results_nice()
        
        # Save std
        task.orig_suffix = orig_orig_suffix + '-std'
        task.results = std
        task.save_results_nice()
        
        task.data_dir = orig_data_dir
        task.results = orig_results
        task.orig_suffix = orig_orig_suffix
    
    def summarize_examples(self):
        
        print("Example trajectories...")
        net = self.create_net('hindex_cumulative')
        with CrossValidation(net) as cv:
            for _ in cv(load=[0], load_to_db=True):
                net.plot_example_trajectories()
        
        print("Example scatter net...")
        net = self.create_net('sqrt_nc_after')
        with CrossValidation(net) as cv:
            # Need DB for hindex predictor
            for _ in cv(load=[0], load_to_db=True):
                # Simulate exclude = []
                net.suffix += '-' + '-'.join([])
                net.plot_correlation()
                net.plot_correlation_hindex()

        print("Example scatter rf...")
        rf = self.create_rf('sqrt_nc_after')
        with CrossValidation(rf) as cv:
            for _ in cv(load=[0], load_to_db=False):
                # Simulate exclude = []
                rf.suffix += '-' + '-'.join([])
                rf.plot_correlation()

    def __summarize_cv_fluctuations(self, net, results, id):
        columns = ['i', 'years', 'loss', 'r', 'R^2', 'MAPE']
        values = {0: [], 5: [], 10: []}

        for i, result in enumerate(results[id]):
            for additional_epochs in result:
                for data in result[additional_epochs]:
                    # Take only after 10 years
                    if data[0] == 10:
                        values[additional_epochs].append([i] + data)

        for additional_epochs in values:

            if not len(values[additional_epochs]):
                continue

            # Workaround to keep .csv file names from summarize_net() the same
            # as they were before we added random forests
            if '-rf' in net.suffix:
                suffix = net.suffix + '-10years'
            else:
                suffix = id.replace(PredictivityOverTime.__name__,
                                    '-10years')
            if additional_epochs > 0:
                suffix += '-add'+str(additional_epochs)
            filename = os.path.join(
                net.evaluate_dir,
                'cv-fluctuations-' + suffix + '.csv')
            pd.DataFrame(values[additional_epochs],
                         columns=columns).to_csv(filename)

    def summarize_rf(self):

        print("Summarizing random forests")

        # Calculate averages

        # Record fluctuations over different cv splits
        # Only implemented for PredictivityOverTime, but that should be enough
        # for now
        cv_fluctuation_ids = [
            'hindex_cumulative' + PredictivityOverTime.__name__
            ]

        for target, tasks in self.rf_targets.items():
            rf = self.create_rf(target)

            # Create results array
            results = {}

            # Collect all results
            print("Collect...")
            with CrossValidation(rf) as cv:
                for i in cv(num=self.num_cv, load=True, load_to_db=False):
                    for task in tasks(rf):
                        id = target + task.__class__.__name__
                        print(i, id)
                        if id not in results:
                            results[id] = []
                        results[id].append(task.load_results())

            # Do the averaging and save
            print("Do average...")
            for task in tasks(rf):
                id = target + task.__class__.__name__
                print(id)
                self.average_and_save(task, results[id])

            # Record cv flucutations
            print("CV fluctuations...")
            for id in cv_fluctuation_ids:
                if id in results:
                    print(id)
                    self.__summarize_cv_fluctuations(rf, results, id)

    def summarize_net(self):

        print("Summarizing net")

        # Calculate averages

        # Record fluctuations over different cv splits
        # Only implemented for PredictivityOverTime, but that should be enough
        # for now
        cv_fluctuation_ids = [
            'hindex_cumulative' + PredictivityOverTime.__name__,
            'hindex_cumulative' + PredictivityOverTime.__name__ + '-naive',
            'hindex_cumulative' + PredictivityOverTime.__name__ + '-linear'
            ]

        for target, tasks in self.targets.items():
            net = self.create_net(target)

            # Create results array
            results = {}

            # Collect all results
            print("Collect...")
            with CrossValidation(net) as cv:
                for i in cv(num=self.num_cv, load=True, load_to_db=False):
                    for task in tasks(net):
                        id = target + task.__class__.__name__
                        print(i, id)
                        if id not in results:
                            results[id] = []
                        results[id].append(task.load_results())

                    # Summarize naive hindex / linear in time predictor
                    # performance

                    # Trick to use Task logic for naive hindex/linear in time
                    # predictors for both Net and TimeSeriesNet
                    orig_suffix = net.suffix

                    # naive h-index predictor
                    net.suffix = orig_suffix + '-naive'
                    with PredictivityOverTime(net, epochs_steps=0) as task:
                        id = target + task.__class__.__name__ + '-naive'
                        print(i, id)
                        if id not in results:
                            results[id] = []
                        results[id].append(task.load_results())

                    # linear in time predictor
                    net.suffix = orig_suffix + '-linear'
                    with PredictivityOverTime(net, epochs_steps=0) as task:
                        id = target + task.__class__.__name__ + '-linear'
                        print(i, id)
                        if id not in results:
                            results[id] = []
                        results[id].append(task.load_results())

                    # Revert trick
                    net.suffix = orig_suffix

            # Do relative data importance for R^2
            print("Do relative R^2 data importance for hindex_cumulative...")
            for task in tasks(net):
                if not isinstance(task, DataImportanceOverTime):
                    continue
                id = target + task.__class__.__name__
                id_reference = target + PredictivityOverTime.__name__

                print("do", id, id_reference)

                mean = {}
                std = {}

                additional_epochs = 0

                # Calculate mean/std
                for years in range(1, net.predict_after_years+1):
                    mean[years] = {}
                    std[years] = {}
                    for exclude in task.excludes:
                        exclude = '-'.join(exclude)

                        ratios = []
                        for (data, ref_data) in zip(results[id],
                                                    results[id_reference]):
                            # R^2 is at 2
                            value = data[additional_epochs][years][exclude][2]

                            # R^2 is at 3
                            ref_values = ref_data[additional_epochs][years-1]
                            if ref_values[0] != years:
                                raise Exception("years not at [0]?!")
                            ref_value = ref_values[3]

                            ratios.append(value/ref_value)

                        mean[years][exclude] = np.mean(ratios)
                        std[years][exclude] = np.std(ratios)

                # Save to files
                if task.includes is not None:
                    columns = ['years'] + ['R^2_rem/R^2 ' + include[0]
                                        for include in task.includes]
                else:
                    columns = ['years'] + ['R^2_rem/R^2 ' + exclude[0]
                                        for exclude in task.excludes]
                mean_data = [
                    [years] + [mean[years]['-'.join(exclude)]
                               for exclude in task.excludes]
                    for years in mean]
                std_data = [
                    [years] + [std[years]['-'.join(exclude)]
                               for exclude in task.excludes]
                    for years in std]

                filename = 'data-importance-over-time-'+target
                if task.includes is not None:
                    filename += '-include'
                mean_file = os.path.join(net.evaluate_dir,
                                         filename+'-ratio-mean.csv')
                std_file = os.path.join(net.evaluate_dir,
                                        filename+'-ratio-std.csv')
                pd.DataFrame(data=mean_data, columns=columns).to_csv(mean_file)
                pd.DataFrame(data=std_data, columns=columns).to_csv(std_file)

            # Do the averaging and save
            print("Do average...")
            for task in tasks(net):
                id = target + task.__class__.__name__
                print(id)
                self.average_and_save(task, results[id])

            # Trick to use Task logic for naive hindex/linear in time
            # predictors for both Net and TimeSeriesNet
            orig_suffix = net.suffix

            # naive h-index predictor
            net.suffix = orig_suffix + '-naive'
            with PredictivityOverTime(net, epochs_steps=0) as task:
                id = target + task.__class__.__name__ + '-naive'
                print(id)
                self.average_and_save(task, results[id])

            # linear in time predictor
            net.suffix = orig_suffix + '-linear'
            with PredictivityOverTime(net, epochs_steps=0) as task:
                id = target + task.__class__.__name__ + '-linear'
                print(id)
                self.average_and_save(task, results[id])

            # Revert trick
            net.suffix = orig_suffix

            # Record cv flucutations
            print("CV fluctuations...")
            for id in cv_fluctuation_ids:
                if id in results:
                    print(id)
                    self.__summarize_cv_fluctuations(net, results, id)

    def summarize(self):

        # Examples with cv 0
        self.summarize_examples()
        
        # Net
        self.summarize_net()

        # Random forests
        self.summarize_rf()

    def subsets_evaluate(self):
        for target, tasks in self.targets.items():
            net = self.create_net(target)
            with CrossValidation(net) as cv:
                # TODO: Can do this for each cv in an "evaluate" step or so and then
                # let the usual summarize*() do its thing / average / etc.?
                # For now only cv 0
                for _ in cv(load=[0], load_to_db=True):
                    for task in tasks(net):
                        task.subsets_evaluate()


class TimeSeriesRunner(Runner):
    
    def __init__(self, net_prepare_func=None):
        super().__init__(net_prepare_func)
        
        self.Net = TimeSeriesNet
        self.net_params['force_monotonic'] = True


if __name__ == '__main__':
    """
    # runner = Runner()
    runner = TimeSeriesRunner()
    runner.num_cv = 2
    
    # runner.prepare()
    # runner.train_one(0)
    # runner.train_one(1)
    runner.evaluate()
    """

    runner = TimeSeriesRunner()
    runner.subsets_evaluate()
