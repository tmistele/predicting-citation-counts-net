import numpy as np
from db import db
from MySQLdb import OperationalError, ProgrammingError, InterfaceError
import settings


class CrossValidation:

    def __init__(self, net):
        self.net = net
        self._c = None

    @property
    def c(self):
        if self._c is None:
            self._c = db().cursor()
        return self._c

    def __enter__(self):

        # Save suffix
        self.orig_suffix = self.net.suffix
        self.orig_suffix_author_ids = self.net.suffix_author_ids

        return self.generator

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Reset suffix
        self.net.suffix = self.orig_suffix
        self.net.suffix_author_ids = self.orig_suffix_author_ids

        # Reset train values
        try:
            self.c.execute("""
                UPDATE analysis{0}_authors
                SET train = train_real""".format(self.net.suffix_cuts))
            db().commit()
        except (OperationalError, InterfaceError):
            pass

    def __load_author_ids(self):

        sql = """
            INSERT INTO analysis{0}_authors
            (author_id, train, first_paper_date)
            VALUES """.format(self.net.suffix_cuts)

        authors = self.net.get_train_authors(generate_if_not_exists=False)

        is_first = True
        for author_id, train in authors:
            if train == -1:
                train = 0

            if is_first:
                is_first = False
            else:
                sql += ", "

            sql += "(%s, %s, '1970-01-01') " % (author_id, train)

        sql += "ON DUPLICATE KEY UPDATE train = VALUES(train)"

        db().cursor().execute(sql)
        db().commit()

    def generator(self, num=None, load=None, load_to_db=True):

        # We use load is None to detect "dont' load", but also allow load=False
        if load is False:
            load = None

        if type(load) is int:
            nums = [load]
        elif type(load) is list:
            nums = load
        elif num is not None:
            nums = range(0, num)
        else:
            raise ValueError("Either num or load need to be set")

        if load_to_db or load is None:
            # Refuse to run bevore train_real is set!
            self.c.execute("""
                SELECT COUNT(*) FROM analysis{0}_authors
                WHERE train_real IS NULL
                """.format(self.net.suffix_cuts))
            if self.c.fetchone()[0] > 0:
                raise Exception("Please set train_real first!")

        for i in nums:
            # Set suffix
            self.net.suffix = self.orig_suffix + '-cv-' + str(i)
            self.net.suffix_author_ids = self.orig_suffix_author_ids +\
                '-cv-' + str(i)

            if load is not None:
                # Load saved training / validation authors
                if load_to_db:
                    self.__load_author_ids()
                else:
                    # To prevent shooting outselves in the foot
                    try:
                        db().close()
                    except (OperationalError, ProgrammingError):
                        pass
                    settings.DB_PASS = '__dont_use_db_with_load_to_db_false__'
            else:

                # Prevent ourselves from accidentally overwriting cv author_ids
                try:
                    self.net.get_train_authors(generate_if_not_exists=False)
                    raise Exception(
                        "cv author_ids already exist! Please remove "
                        "author_ids file if you want to generate new ids."
                        "Otherwise, use load=True")
                except FileNotFoundError:
                    pass

                # Choose new training authors
                self.c.execute("""
                    UPDATE analysis{0}_authors
                    SET train = NULL""".format(self.net.suffix_cuts))
                self.net.choose_train_authors()
                db().commit()

            yield i


def crossvalidate_test(net):
    with CrossValidation(net) as cv:

        c = db().cursor()

        r = []
        r2 = []
        manypapers = []

        if net.cutoff != net.CUTOFF_SINGLE:
            raise Exception("#papers stuff not implemented!")

        num = 20

        find_bad_authors = False
        if find_bad_authors:
            i = 0
            authors = []
            runs = np.zeros(num)

        find_ratio2 = False
        if find_ratio2:
            import pickle
            (backend_author_ids, _) = pickle.load(open(
                os.path.join(settings.DATA_DIR, 'arxiv',
                             'keywords-backend', 'author_ids'), 'rb'))
            author_names = {}
            for name, id in backend_author_ids.items():
                if id not in author_names:
                    author_names[id] = []
                author_names[id].append(name)
            backend_author_ids = None
            ratio2 = []

        find_zero = False
        if find_zero:
            zero = []

        find_lowhindex = False
        if find_lowhindex:
            lowhindex = []

        for _ in cv(num=num, load=True):

            # Hack to enable setting train = NULL in db
            # Enable if needed
            # os.remove(net.get_author_ids_filename())

            # Exclude people with #papers > 120
            # c.execute("""
            #    UPDATE analysis{0}_authors AS a
            #    SET a.train = NULL
            #    WHERE (
            #        SELECT COUNT(*)
            #       FROM `analysis{0}_fast_paper_authors` AS pa
            #       WHERE
            #       pa.author_id = a.author_id AND
            #       pa.date_created <= %(cutoff_date)s
            #       ) > %(max_papers)s""".format(net.suffix_cuts), {
            #       'cutoff_date': net.cutoff_date,
            #       'max_papers': 120
            #   })

            # Exclude people with no papers after cutoff
            # c.execute("""
            #     UPDATE analysis{0}_authors AS a
            #     SET a.train = NULL
            #     WHERE 0 = (
            #        SELECT COUNT(*)
            #        FROM `analysis{0}_fast_paper_authors` AS pa
            #        WHERE
            #        pa.author_id = a.author_id AND
            #        pa.date_created >= %(cutoff_date)s
            #        )""".format(net.suffix_cuts), {
            #        'cutoff_date': net.cutoff_date
            #   })

            # Exclude people with high broadness
            # c.execute("""
            #     UPDATE analysis{0}_authors AS a
            #     SET a.train = NULL
            #     WHERE a.broadness_lda > 3""".format(net.suffix_cuts), {
            #        'cutoff_date': net.cutoff_date
            #    })

            # Exclude people with zero hinex_cumulative
            # c.execute("""
            #     UPDATE analysis{0}_authors AS a
            #     INNER JOIN analysis{0}_hindex_data AS h
            #     ON h.author_id = a.author_id AND
            #         h.predict_after_years = %(predict_after_years)s AND
            #         h.hindex_cumulative = 0
            #     SET a.train = NULL
            #     """.format(net.suffix_cuts), {
            #         'predict_after_years': net.predict_after_years})

            # Naive hindex predictor
            _, r_hindex, r2_hindex, mae_hindex = net.hindex_evaluate()
            r.append(r_hindex)
            r2.append(r2_hindex)
            print("r ", r_hindex)
            print("R^2 ", r2_hindex)

            if find_bad_authors:
                runs[i] = r2_hindex
                c.execute("""
                    SELECT author_id
                    FROM analysis{0}_authors
                    WHERE train = 0""".format(net.suffix_cuts))
                authors.append(c.fetchall())

                i += 1

            if find_ratio2:
                numauthors = c.execute("""
                    SELECT author_id
                    FROM analysis{0}_authors
                    WHERE train = 0""".format(net.suffix_cuts))

                num2 = 0
                for (author_id,) in c:
                    if len(author_names[author_id]) <= 2:
                        num2 += 1.

                ratio2.append(num2/numauthors + .3)

            if find_zero:
                numauthors = c.execute("""
                    SELECT a.author_id
                    FROM analysis{0}_authors AS a
                    INNER JOIN analysis{0}_hindex_data AS h
                    ON h.author_id = a.author_id AND
                        h.predict_after_years = %(predict_after_years)s AND
                        h.hindex_cumulative = 0
                    WHERE a.train = 0""".format(net.suffix_cuts), {
                        'predict_after_years': net.predict_after_years})

                zero.append(numauthors / 700. + .5)

            if find_lowhindex:
                numauthors = c.execute("""
                    SELECT a.author_id
                    FROM analysis{0}_authors AS a
                    INNER JOIN analysis{0}_hindex_data AS h
                    ON h.author_id = a.author_id AND
                        h.predict_after_years = %(predict_after_years)s AND
                        h.hindex_before < 4
                    WHERE a.train = 0""".format(net.suffix_cuts), {
                        'predict_after_years': net.predict_after_years})

                lowhindex.append(numauthors/7000. - .25)

        if find_bad_authors:
            worst_runs = runs.argsort()[:int(np.ceil(num/10.))]

            bad_authors = np.intersect1d(authors[worst_runs[0]],
                                         authors[worst_runs[1]])
            for i in range(2, len(worst_runs)):
                bad_authors = np.intersect1d(bad_authors,
                                             authors[worst_runs[i]])

            print("bad authors", bad_authors)
            print("# bad authors", len(bad_authors))

        import matplotlib.pyplot as plt

        if find_ratio2:
            plt.plot(ratio2, label="ratio2")
        if find_zero:
            plt.plot(zero, label="zero hindex")
        if find_lowhindex:
            plt.plot(lowhindex, label="low hindex")

        plt.plot(r, label="r naive")
        plt.plot(r2, label="R^2 naive")
        plt.legend()
        plt.show()
        plt.close()

        r2mean = sum(r2)/len(r2)

        print("All r", r)
        print("All R^2", r2)
        if find_ratio2:
            print("All ratio2", ratio2)
            ratio2mean = sum(ratio2)/len(ratio2)
            print(np.corrcoef(
                np.array(ratio2) - ratio2mean,
                np.array(r2) - r2mean
                ))
        if find_zero:
            print("All zero", zero)
            zeromean = sum(zero)/len(zero)
            print(np.corrcoef(
                np.array(zero) - zeromean,
                np.array(r2) - r2mean
                ))
        if find_lowhindex:
            print("All low hindex", lowhindex)
            lowmean = sum(lowhindex)/len(lowhindex)
            print(np.corrcoef(
                np.array(lowhindex) - lowmean,
                np.array(r2) - r2mean
                ))
        print("R^2 avg", r2mean)
        print("R^2 std", np.std(r2))
        print("R^2 amplitude", max(r2) - min(r2))
