from abc import ABC, abstractmethod
from db import db
from datetime import date
import numpy as np

class SubsetsEvaluation(ABC):

    default_runs = []

    def __init__(self, net, runs=None):
        if runs is None:
            runs = self.default_runs
        self.net = net
        self.runs = runs

    @abstractmethod
    def get_author_ids(self, params):
        return []

    def run(self):
        orig_pick_validation_authors = \
            self.net.__class__.pick_validation_authors

        def fake_pick_validation_authors(net, authors):
            indices = orig_pick_validation_authors(net, authors)

            author_ids = self.get_author_ids(params)
            print(len(indices[0]))
            indices = (np.intersect1d(
                np.where(np.isin(authors['author_id'], author_ids)),
                indices),)
            print(params, len(indices[0]))
            return indices
        self.net.__class__.pick_validation_authors = \
            fake_pick_validation_authors

        results = []
        for params in self.runs:
            result = self.net.evaluate()
            print(params, "\n")
            results.append(result)

        self.net.__class__.pick_validation_authors = \
            orig_pick_validation_authors

        return results


class FirstPaperDateSubsetsEvaluation(SubsetsEvaluation):

    default_runs = [
        (date(1996, 1, 1), date(1997, 1, 1)),
        (date(1997, 1, 1), date(1998, 1, 1)),
        (date(1998, 1, 1), date(1999, 1, 1)),
        (date(1999, 1, 1), date(2000, 1, 1)),
        (date(2000, 1, 1), date(2001, 1, 1)),
        (date(2001, 1, 1), date(2002, 1, 1)),
        (date(2002, 1, 1), date(2003, 1, 1))
        ]

    def get_author_ids(self, params):
        (min_first_paper_date, max_first_paper_date) = params
        sql = """
            SELECT a.author_id
            FROM analysis{0}_authors AS a

            WHERE
            a.first_paper_date BETWEEN
                %(min_first_paper_date)s AND %(max_first_paper_date)s
            """.format(self.net.suffix_cuts)
        c = db().cursor()
        numauthors = c.execute(sql, {
            'min_first_paper_date': min_first_paper_date,
            'max_first_paper_date': max_first_paper_date})
        return np.fromiter(
            c, count=numauthors, dtype=[('author_id', 'i4')])['author_id']


class HindexCumulativeSubsetsEvaluation(SubsetsEvaluation):

    default_runs = [
        (0, 60),
        (0, 5),
        (5, 10),
        (10, 20),
        (20, 40),
        (40, 60)
        ]

    hindex_field = 'hindex_cumulative'

    def get_author_ids(self, params):
        (min_hindex, max_hindex) = params
        sql = """
            SELECT a.author_id
            FROM analysis{0}_authors AS a

            INNER JOIN analysis{0}_hindex_data AS h
            ON h.author_id = a.author_id

            WHERE
            h.predict_after_years = 10 AND
            h.{1}
                BETWEEN %(min_hindex)s AND %(max_hindex)s

            """.format(self.net.suffix_cuts, self.hindex_field)
        c = db().cursor()
        numauthors = c.execute(sql, {
            'min_hindex': min_hindex,
            'max_hindex': max_hindex})
        return np.fromiter(
            c, count=numauthors, dtype=[('author_id', 'i4')])['author_id']


class HindexBeforeSubsetsEvaluation(HindexCumulativeSubsetsEvaluation):
    default_runs = [
        (0, 45),
        (0, 22),
        (22, 45),
        (0, 10),
        (10, 22),
        (0, 5),
        (5, 10)
    ]

class SqrtNcAfterSubsetsEvaluation(SubsetsEvaluation):
    default_runs = [
        (0, 99),
        (0, 10000),
        ]

    def get_author_ids(self, params):
        (min_sqrt_nc, max_sqrt_nc) = params
        sql = """
            SELECT a.author_id
            FROM analysis{0}_authors AS a

            INNER JOIN analysis{0}_nc_data AS n
            ON n.author_id = a.author_id

            WHERE
            n.predict_after_years = 10 AND
            SQRT(n.nc_after)
                BETWEEN %(min_sqrt_nc)s AND %(max_sqrt_nc)s

            """.format(self.net.suffix_cuts)
        c = db().cursor()
        numauthors = c.execute(sql, {
            'min_sqrt_nc': min_sqrt_nc,
            'max_sqrt_nc': max_sqrt_nc})
        return np.fromiter(
            c, count=numauthors, dtype=[('author_id', 'i4')])['author_id']

