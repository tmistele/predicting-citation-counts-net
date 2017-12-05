import settings
import os
from db import db
from datetime import date
from dateutil.relativedelta import relativedelta
import numpy as np


class Analysis:
    
    default_first_paper_range = (5, 12)
    
    max_papers = 500
    max_authors = 30
    
    split_after_years = 10
    predict_after_years = 10
    
    cutoff_date = date(2008, 1, 1)
    
    CUTOFF_SINGLE = 'single'
    CUTOFF_PERAUTHOR = ''
    
    def __init__(self, cutoff, first_paper_range=None):
        
        # Set cutoff type
        # NOTE: Data from keyword analysis (topics, broadness) is generated
        #       with CUTOFF_SINGLE in mind. Therefore, CUTOFF_PERAUTHOR makes
        #       sense only when not using any of this data
        if 'perauthor' == cutoff:
            cutoff = self.CUTOFF_PERAUTHOR
        self.cutoff = cutoff
        if self.cutoff == self.CUTOFF_PERAUTHOR:
            self.get_split_date = self.get_split_date_perauthor
            self.get_end_date = self.get_end_date_perauthor
        elif self.cutoff == self.CUTOFF_SINGLE:
            self.get_split_date = self.get_split_date_single
            self.get_end_date = self.get_end_date_single
        else:
            raise Exception("Not implemented")
        
        # Min/max paper date
        if first_paper_range is None:
            self.first_paper_range = self.default_first_paper_range
        else:
            self.first_paper_range = first_paper_range
        self.first_paper_max_date = self.cutoff_date - \
            relativedelta(years=self.first_paper_range[0])
        self.first_paper_min_date = self.cutoff_date - \
            relativedelta(years=self.first_paper_range[1])
        
        # Suffix indicating author cuts
        self.suffix_cuts = self.cutoff +\
            ''.join([str(x) for x in self.first_paper_range])
        
        # Set data dir
        self.data_dir = os.path.join(settings.DATA_DIR,
                                     'analysis{0}'.format(self.suffix_cuts))
        
        # Directory for evaluation results
        self.evaluate_dir = os.path.join(self.data_dir, 'evaluate')
        
        # Make sure directories exist
        os.makedirs(self.evaluate_dir, exist_ok=True)
    
    def get_split_date_single(self, first_paper_date):
        return self.cutoff_date
    
    def get_end_date_single(self, first_paper_date):
        return self.cutoff_date + relativedelta(years=self.predict_after_years)
    
    def get_split_date_perauthor(self, first_paper_date):
        return first_paper_date + relativedelta(years=self.split_after_years)
    
    def get_end_date_perauthor(self, first_paper_date):
        return first_paper_date + relativedelta(
            years=self.split_after_years+self.predict_after_years)
    
    def generate_authors(self):
        
        print("Building analysis_authors...")
        
        # Select authors which have published the first paper during 1994-1998
        # Do not select authors with only 1 or 2 papers or more than 500
        # papers. Do not include papers with more than 30 authors.
        c = db().cursor()
        c.execute(
            """
            INSERT INTO analysis{0}_authors
            (author_id, first_paper_date, broadness, broadness_lda)
            
            SELECT a.id, MIN(p.date_created), a.broadness, a.broadness_lda
            FROM `authors` AS a
            
            INNER JOIN paper_authors AS pa
            ON pa.author_id = a.id
            INNER JOIN papers AS p
            ON p.id = pa.paper_id AND p.num_authors < %(max_authors)s
            
            WHERE
            a.name NOT LIKE '%%collaboration%%'
            
            GROUP BY a.id
            HAVING
            COUNT(*) >= 5 AND COUNT(*) < %(max_papers)s AND
            MIN(p.date_created) >= %(min_first_paper_date)s AND
            MIN(p.date_created) < %(max_first_paper_date)s
            """.format(self.suffix_cuts), {
                'max_papers': self.max_papers,
                'max_authors': self.max_authors,
                'min_first_paper_date': self.first_paper_min_date,
                'max_first_paper_date': self.first_paper_max_date,
                })
        
        print("Committing")
        db().commit()
    
    # Generates data for authors in analysis_authors
    # already joined together in a useful way for later analysis
    def generate_fast_tables(self):
        c = db().cursor()

        print("Building analysis_fast_paper_authors...")
        if self.cutoff == self.CUTOFF_PERAUTHOR:
            c.execute("""
                INSERT INTO analysis{0}_fast_paper_authors
                    (paper_id, author_id, date_created, length, jif, published,
                    journal, country)
                
                SELECT
                    pa.paper_id, pa.author_id, p.date_created, p.length,
                    IFNULL(jif.jif, 0),
                    IF(p.journal IS NULL OR p.journal = '', 0, 1),
                    p.journal, pa.country
                FROM paper_authors AS pa
                INNER JOIN analysis{0}_authors AS a
                ON a.author_id = pa.author_id
                INNER JOIN papers AS p
                ON p.id = pa.paper_id AND p.num_authors < %(max_authors)s
                LEFT JOIN jif AS jif
                ON jif.journal = p.journal AND
                jif.year = YEAR(DATE_ADD(
                            a.first_paper_date,
                            INTERVAL %(split_after_years)s YEAR))-1
                """.format(self.suffix_cuts), {
                    'max_authors': self.max_authors,
                    'split_after_years': self.split_after_years})
        elif self.cutoff == self.CUTOFF_SINGLE:
            c.execute("""
                INSERT INTO analysis{0}_fast_paper_authors
                    (paper_id, author_id, date_created, length, jif, published,
                    journal, country)
                
                SELECT
                    pa.paper_id, pa.author_id, p.date_created, p.length,
                    IFNULL(jif.jif, 0),
                    IF(p.journal IS NULL OR p.journal = '', 0, 1),
                    p.journal, pa.country
                FROM paper_authors AS pa
                INNER JOIN analysis{0}_authors AS a
                ON a.author_id = pa.author_id
                INNER JOIN papers AS p
                ON p.id = pa.paper_id AND p.num_authors < %(max_authors)s
                LEFT JOIN jif AS jif
                ON jif.journal = p.journal AND
                jif.year = %(cutoff_year)s-1
                """.format(self.suffix_cuts), {
                    'max_authors': self.max_authors,
                    'cutoff_year': self.cutoff_date.year})
        else:
            raise Exception("Not implemented")
        
        print("Building analysis_fast_citations...")
        c.execute("""
            INSERT INTO analysis{0}_fast_citations
                (cited_paper, citing_paper,
                cited_paper_date_created, citing_paper_date_created)
            
            SELECT p.id, c.citing_paper,
                p.date_created, cp.date_created
            FROM analysis{0}_authors AS a
            
            INNER JOIN paper_authors AS pa
            ON pa.author_id = a.author_id
            
            INNER JOIN papers AS p
            ON p.id = pa.paper_id AND p.num_authors < %(max_authors)s
            
            INNER JOIN citations AS c
            ON c.cited_paper = p.id
            
            INNER JOIN papers AS cp
            ON cp.id = c.citing_paper AND cp.num_authors < %(max_authors)s
            
            GROUP BY c.citing_paper, c.cited_paper""".format(self.suffix_cuts), {
                'max_authors': self.max_authors})
        
        print("Building analysis_fast_coauthors...")
        c.execute("""
            INSERT INTO analysis{0}_fast_coauthors
                (analysis_author_id, coauthor_id, first_date)
            
            SELECT apa.author_id AS analysis_author_id,
                pa.author_id AS coauthor_id,
                MIN(apa.date_created) AS first_date
            FROM analysis{0}_fast_paper_authors AS apa
            
            INNER JOIN paper_authors AS pa
            ON pa.paper_id = apa.paper_id
                AND pa.author_id != apa.author_id
            
            GROUP BY apa.author_id, pa.author_id
            """.format(self.suffix_cuts))
        
        print("Committing")
        db().commit()
    
    def __get_nc_of_author_fast(self, author_id, start_date, end_date):
        sql = """ SELECT COUNT(*) AS num_citations
            
            FROM analysis{0}_fast_citations AS c
            
            INNER JOIN paper_authors AS pa
            ON pa.paper_id = c.cited_paper AND pa.author_id = %(author_id)s
            
            WHERE
            c.cited_paper_date_created >= %(start_date)s
            AND c.cited_paper_date_created < %(end_date)s
            AND c.citing_paper_date_created >= %(start_date)s
            AND c.citing_paper_date_created < %(end_date)s
            
            """.format(self.suffix_cuts)
        params = {
            'author_id': author_id,
            'start_date': start_date,
            'end_date': end_date
        }
        
        c = db().cursor()
        c.execute(sql, params)
        
        return c.fetchone()[0]
    
    def __generate_nc_data_single(self):
        
        print("Generating nc data (single)...")
        
        c = db().cursor()
        
        # Generate data
        sql = """
            SELECT
            COUNT(*), pa.author_id
            FROM analysis{0}_fast_paper_authors AS pa
            
            INNER JOIN analysis{0}_fast_citations AS c
            ON pa.paper_id = c.cited_paper AND
                c.citing_paper_date_created < %(end_date)s AND
                c.citing_paper_date_created >= %(start_date)s
            
            WHERE
            pa.date_created < %(end_date)s AND
            pa.date_created >= %(start_date)s
            
            GROUP BY pa.author_id
            """.format(self.suffix_cuts)
        
        result = {}
        
        runs = [
            ('nc_before',
             self.first_paper_min_date,
             self.get_split_date(None)),
            ('nc_after',
             self.get_split_date(None),
             self.get_end_date(None)),
            ('nc_cumulative',
             self.first_paper_min_date,
             self.get_end_date(None))
            ]
        
        for i, (field, start_date, end_date) in enumerate(runs):
            print(field)
            c.execute(sql, {
                    'start_date': start_date,
                    'end_date': end_date})
            
            for (num_citations, author_id) in c:
                
                print("nc:", author_id, num_citations)
                if author_id not in result:
                    result[author_id] = [0 for _ in runs]
                result[author_id][i] = num_citations
        
        # Add authors which don't have *any* citations
        c.execute(("SELECT author_id FROM analysis{0}_authors " +
                   "WHERE author_id NOT IN(" +
                   ' , '.join([str(x) for x in result]) +
                   ")").format(self.suffix_cuts))
        for (author_id,) in c:
            result[author_id] = [0 for _ in runs]
 
        # Write to database
        fields = ', '.join(['author_id', 'predict_after_years'] +
                           [field for field, _, _ in runs])
        values = '), ('.join([
            ' , '.join([str(author_id), str(self.predict_after_years)] +
                       [str(x) for x in hindex])
            for author_id, hindex in result.items()])

        sql = ("INSERT INTO analysis{0}_nc_data ("+fields+") VALUES (" +
               values + ")").format(self.suffix_cuts)
        c.execute(sql)
        db().commit()
    
    def generate_nc_data(self):
        
        print("Generating nc_data...")
        c2 = db().cursor()
        
        c2.execute("""
            SELECT COUNT(*)
            FROM analysis{0}_nc_data
            WHERE predict_after_years = %(predict_after_years)s
            """.format(self.suffix_cuts), {
                'predict_after_years': self.predict_after_years})
        if c2.fetchone()[0] > 0:
            print("-> Already exists, skipping")
            return
        
        if self.cutoff == self.CUTOFF_SINGLE:
            # Optimized version for single cutoff
            return self.__generate_nc_data_single()
        
        c = db().cursor()
        c.execute("""SELECT author_id, first_paper_date
            FROM analysis{0}_authors""".format(self.suffix_cuts))
        for row in c:
            author_id = row[0]
            first_paper_date = row[1]
            split_date = self.get_split_date(first_paper_date)
            end_date = self.get_end_date(first_paper_date)
            
            nc_before = self.__get_nc_of_author_fast(
                author_id,
                start_date=first_paper_date, end_date=split_date)
            nc_after = self.__get_nc_of_author_fast(
                author_id,
                start_date=split_date, end_date=end_date)
            nc_cumulative = self.__get_nc_of_author_fast(
                author_id,
                start_date=first_paper_date, end_date=end_date)
            
            c2.execute(
                """
                INSERT INTO analysis{0}_nc_data SET
                author_id=%(author_id)s,
                predict_after_years=%(predict_after_years)s,
                nc_before=%(nc_before)s,
                nc_after=%(nc_after)s,
                nc_cumulative=%(nc_cumulative)s
                """.format(self.suffix_cuts),
                {
                    'author_id': author_id,
                    'predict_after_years': self.predict_after_years,
                    'nc_before': nc_before,
                    'nc_after': nc_after,
                    'nc_cumulative': nc_cumulative
                }
            )
            
            print(author_id, nc_before, nc_after, nc_cumulative,
                  first_paper_date.strftime("%Y-%m-%d"))
        
        print("Committing")
        db().commit()
    
    def get_hindex_of_author_fast(self, author_id, start_date, end_date):
        return self.__get_hindex_of_author_fast(author_id, start_date,
                                                end_date)
    
    def __get_hindex_of_author_fast(self, author_id, start_date, end_date):
        
        # Fetch ordered list of the number of citations of each paper
        sql = """
            SELECT COUNT(*) AS num_citations
            
            FROM analysis{0}_fast_citations AS c
            
            INNER JOIN paper_authors AS pa
            ON pa.paper_id = c.cited_paper AND pa.author_id = %(author_id)s
            
            WHERE
            c.cited_paper_date_created >= %(start_date)s
            AND c.cited_paper_date_created < %(end_date)s
            AND c.citing_paper_date_created >= %(start_date)s
            AND c.citing_paper_date_created < %(end_date)s

            GROUP BY c.cited_paper
            ORDER BY COUNT(*) DESC
            """.format(self.suffix_cuts)
        params = {
            'author_id': author_id,
            'start_date': start_date,
            'end_date': end_date
        }
        
        c = db().cursor()
        c.execute(sql, params)
        
        # Calculate h-index from this
        hindex = 0
        for row in c:
            if row[0] <= hindex:
                break
            hindex += 1
        
        return hindex

    def __generate_hindex_data_single(self):
        print("Generating hindex data (single)...")
        
        c = db().cursor()
        
        # Generate data
        sql = """
            SELECT
            COUNT(*), pa.author_id
            FROM analysis{0}_fast_paper_authors AS pa
            
            INNER JOIN analysis{0}_fast_citations AS c
            ON pa.paper_id = c.cited_paper AND
                c.citing_paper_date_created < %(end_date)s AND
                c.citing_paper_date_created >= %(start_date)s
            
            WHERE
            pa.date_created < %(end_date)s AND
            pa.date_created >= %(start_date)s
            
            GROUP BY pa.author_id, pa.paper_id
            
            ORDER BY pa.author_id ASC, COUNT(*) DESC
            """.format(self.suffix_cuts)
        
        result = {}
        
        runs = [
            ('hindex_before',
             self.first_paper_min_date,
             self.get_split_date(None)),
            ('hindex_after',
             self.get_split_date(None),
             self.get_end_date(None)),
            ('hindex_cumulative',
             self.first_paper_min_date,
             self.get_end_date(None))
            ]
        
        for i, (field, start_date, end_date) in enumerate(runs):
            print(field)
            c.execute(sql, {
                    'start_date': start_date,
                    'end_date': end_date})
            
            last_author_id = None
            hindex = 0
            values = []
            for (num_citations, author_id) in c:
                
                if author_id != last_author_id and last_author_id is not None:
                    print("h-index:", last_author_id, hindex)
                    if last_author_id not in result:
                        result[last_author_id] = [0 for _ in runs]
                    result[last_author_id][i] = hindex
                        
                    hindex = 0
                
                if num_citations > hindex:
                    hindex += 1
                
                last_author_id = author_id
            
            if last_author_id is not None:
                print("h-index:", last_author_id, hindex)
                if last_author_id not in result:
                    result[last_author_id] = [0 for _ in runs]
                result[last_author_id][i] = hindex
        
        # Add authors which don't have *any* citations
        c.execute(("SELECT author_id FROM analysis{0}_authors " +
                   "WHERE author_id NOT IN(" +
                   ' , '.join([str(x) for x in result]) +
                   ")").format(self.suffix_cuts))
        for (author_id,) in c:
            result[author_id] = [0 for _ in runs]
 
        # Write to database
        fields = ', '.join(['author_id', 'predict_after_years'] +
                           [field for field, _, _ in runs])
        values = '), ('.join([
            ' , '.join([str(author_id), str(self.predict_after_years)] +
                       [str(x) for x in hindex])
            for author_id, hindex in result.items()])

        sql = ("INSERT INTO analysis{0}_hindex_data ("+fields+") VALUES (" +
               values + ")").format(self.suffix_cuts)
        c.execute(sql)
        db().commit()
    
    def generate_hindex_data(self):
        
        print("Generating hindex data...")
        
        c2 = db().cursor()
        
        c2.execute("""
            SELECT COUNT(*)
            FROM analysis{0}_hindex_data
            WHERE predict_after_years = %(predict_after_years)s
            """.format(self.suffix_cuts), {
                'predict_after_years': self.predict_after_years})
        if c2.fetchone()[0] > 0:
            print("-> Already exists, skipping")
            return
            
        if self.cutoff == self.CUTOFF_SINGLE:
            # Optimized version for single cutoff
            return self.__generate_hindex_data_single()
        
        c = db().cursor()
        c.execute("""SELECT author_id, first_paper_date
            FROM analysis{0}_authors""".format(self.suffix_cuts))
        for row in c:
            author_id = row[0]
            first_paper_date = row[1]
            split_date = self.get_split_date(first_paper_date)
            end_date = self.get_end_date(first_paper_date)
            
            hindex_before = self.__get_hindex_of_author_fast(
                author_id,
                start_date=first_paper_date, end_date=split_date)
            hindex_after = self.__get_hindex_of_author_fast(
                author_id,
                start_date=split_date, end_date=end_date)
            hindex_cumulative = self.__get_hindex_of_author_fast(
                author_id,
                start_date=first_paper_date, end_date=end_date)
            
            c2.execute(
                """
                INSERT INTO analysis{0}_hindex_data SET
                author_id=%(author_id)s,
                predict_after_years=%(predict_after_years)s,
                hindex_before=%(hindex_before)s,
                hindex_after=%(hindex_after)s,
                hindex_cumulative=%(hindex_cumulative)s
                """.format(self.suffix_cuts),
                {
                    'author_id': author_id,
                    'predict_after_years': self.predict_after_years,
                    'hindex_before': hindex_before,
                    'hindex_after': hindex_after,
                    'hindex_cumulative': hindex_cumulative
                }
            )
            
            print(author_id, hindex_before, hindex_after, hindex_cumulative,
                  first_paper_date.strftime("%Y-%m-%d"))
        
        print("Committing")
        db().commit()
    
    def __load_plot_data(self):
        c = db().cursor()
       
        # Load nc and hindex data into numpy arrays
        numauthors = c.execute("""
            SELECT nc_before, nc_after, nc_cumulative
            FROM analysis{0}_nc_data
            WHERE predict_after_years = %(predict_after_years)s
            ORDER BY author_id ASC""".format(self.suffix_cuts), {
                'predict_after_years': self.predict_after_years})
        nc_sqrt_data = np.fromiter(
            c.fetchall(), count=numauthors,
            dtype=[('nc_before', 'f4'),
                   ('nc_after', 'f4'),
                   ('nc_cumulative', 'f4')])
        nc_sqrt_data['nc_before'] = np.sqrt(nc_sqrt_data['nc_before'])
        nc_sqrt_data['nc_after'] = np.sqrt(nc_sqrt_data['nc_after'])
        nc_sqrt_data['nc_cumulative'] = np.sqrt(nc_sqrt_data['nc_cumulative'])
         
        numauthors = c.execute("""
            SELECT hindex_before, hindex_after, hindex_cumulative
            FROM analysis{0}_hindex_data
            WHERE predict_after_years = %(predict_after_years)s
            ORDER BY author_id ASC""".format(self.suffix_cuts), {
                'predict_after_years': self.predict_after_years})
        hindex_data = np.fromiter(
            c.fetchall(), count=numauthors,
            dtype=[('hindex_before', 'i4'),
                   ('hindex_after', 'i4'),
                   ('hindex_cumulative', 'i4')])
        
        return hindex_data, nc_sqrt_data
    
    def plot_ydistributions(self):
        hindex_data, nc_sqrt_data = self.__load_plot_data()
        
        import matplotlib.pyplot as plt
        
        plt.subplot(121)
        plt.title("hindex_cumulative")
        plt.hist(hindex_data['hindex_cumulative'], 50)
        
        plt.subplot(122)
        plt.title("sqrt_nc_after")
        plt.hist(nc_sqrt_data['nc_after'], 100)
        
        plt.show()
    
    def plot_hirsch(self):
        
        hindex_data, nc_sqrt_data = self.__load_plot_data()
        
        import csv
        
        def export_to_dat(name, d1, d2):
            name = name + '-' + d1 + '-' + d2
            if d1 in hindex_data.dtype.names:
                d1 = hindex_data[d1]
            elif d1 in nc_sqrt_data.dtype.names:
                d1 = nc_sqrt_data[d1]
            else:
                raise Exception("Invalid d1 %s" % d1)
            if d2 in hindex_data.dtype.names:
                d2 = hindex_data[d2]
            elif d2 in nc_sqrt_data.dtype.names:
                d2 = nc_sqrt_data[d2]
            else:
                raise Exception("Invalid d2 %s" % d2)
            file = os.path.join(self.evaluate_dir,
                                'hirsch-plot-data-%s.dat' % name)
            writer = csv.writer(open(file, 'w'), delimiter=' ')
            writer.writerows(zip(d1, d2))
        
        # NOTE: This actually only makes sense for CUTOFF_PERAUTHOR since
        #       that's what Hirsch originally used!
       
        import matplotlib.pyplot as plt
        plt.rc('text', usetex=True)
        
        # Plots similar to Figs. 2a/b in Hirsch
        ax = plt.subplot(321)
        plt.title("~ Fig. 2a, 9a Hirsch")
        plt.scatter(hindex_data['hindex_before'],
                    nc_sqrt_data['nc_cumulative'], s=10, alpha=.02)
        plt.xlabel(r"$h(t_1)$")
        plt.ylabel(r"$N_{\textrm{c}}(t_2)^{1/2}$")
        
        r = np.corrcoef(hindex_data['hindex_before'],
                        nc_sqrt_data['nc_cumulative'])[1, 0]
        plt.text(0.8, 0.8, 'r=%.2f' % r, transform=ax.transAxes)
        export_to_dat('2a9a', 'hindex_before', 'nc_cumulative')
        
        ax = plt.subplot(322)
        plt.title("~ Fig. 2b, 9b Hirsch")
        plt.scatter(nc_sqrt_data['nc_before'],
                    nc_sqrt_data['nc_cumulative'], s=10, alpha=.02)
        plt.xlabel(r"$N_\textrm{c}(t_1)^{1/2}$")
        plt.ylabel(r"$N_\textrm{c}(t_2)^{1/2}$")
        
        r = np.corrcoef(nc_sqrt_data['nc_before'],
                        nc_sqrt_data['nc_cumulative'])[1, 0]
        plt.text(0.8, 0.8, 'r=%.2f' % r, transform=ax.transAxes)
        export_to_dat('2b9b', 'nc_before', 'nc_cumulative')
       
        # Plots similar to Figs. 3a,5a in Hirsch
        ax = plt.subplot(323)
        plt.title("~ Fig. 5a, 8a Hirsch")
        plt.scatter(hindex_data['hindex_before'],
                    hindex_data['hindex_after'], s=10, alpha=.02)
        plt.xlabel(r"$h(t_1)$")
        plt.ylabel(r"$h(t_1, t_2)$")
        
        r = np.corrcoef(hindex_data['hindex_before'],
                        hindex_data['hindex_after'])[1, 0]
        plt.text(0.8, 0.8, 'r=%.2f' % r, transform=ax.transAxes)
        export_to_dat('5a8a', 'hindex_before', 'hindex_after')
        
        ax = plt.subplot(324)
        plt.title("~ Fig. 3a, 10a Hirsch")
        plt.scatter(hindex_data['hindex_before'],
                    hindex_data['hindex_cumulative'], s=10, alpha=.02)
        # plt.plot([0, 50], [0, 50], 'r')
        plt.xlabel(r"$h(t_1)$")
        plt.ylabel(r"$h(t_2)$")
        
        r = np.corrcoef(hindex_data['hindex_before'],
                        hindex_data['hindex_cumulative'])[1, 0]
        plt.text(0.8, 0.8, 'r=%.2f' % r, transform=ax.transAxes)
        export_to_dat('3a10a', 'hindex_before', 'hindex_cumulative')
        
        # Plots similar to Figs. 4a/b in Hirsch
        ax = plt.subplot(325)
        plt.title("~ Fig. 4a, 7a Hirsch")
        plt.scatter(hindex_data['hindex_before'],
                    nc_sqrt_data['nc_after'], s=10, alpha=.02)
        plt.xlabel(r"$h(t_1)$")
        plt.ylabel(r"$N_\textrm{c}(t_1, t_2)^{1/2}$")
        
        r = np.corrcoef(hindex_data['hindex_before'],
                        nc_sqrt_data['nc_after'])[1, 0]
        plt.text(0.8, 0.8, 'r=%.2f' % r, transform=ax.transAxes)
        export_to_dat('4a7a', 'hindex_before', 'nc_after')
        
        ax = plt.subplot(326)
        plt.title("~ Fig. 4b, 7b Hirsch")
        plt.scatter(nc_sqrt_data['nc_before'],
                    nc_sqrt_data['nc_after'], s=10, alpha=.02)
        plt.xlabel(r"$N_\textrm{c}(t_1)^{1/2}$")
        plt.ylabel(r"$N_\textrm{c}(t_1, t_2)^{1/2}$")
        
        r = np.corrcoef(nc_sqrt_data['nc_before'],
                        nc_sqrt_data['nc_after'])[1, 0]
        plt.text(0.8, 0.8, 'r=%.2f' % r, transform=ax.transAxes)
        export_to_dat('4b7b', 'nc_before', 'nc_after')
        
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.evaluate_dir, 'plot_hirsch-%s.png' %
                                 self.predict_after_years))
        # plt.show()
        plt.close()
    
    def acuna_export_data(self):
        c = db().cursor()
        
        sql = """
            SELECT

            hi.hindex_cumulative,
            
            SQRT(nc.nc_before),
            hi.hindex_before,
            DATEDIFF(%(cutoff_date)s, aa.first_paper_date)/365,
            (
                SELECT COUNT(DISTINCT pa.journal)
                FROM analysis{0}_fast_paper_authors AS pa
                WHERE
                pa.author_id = aa.author_id AND
                pa.date_created < %(cutoff_date)s AND
                pa.journal != '' AND
                pa.journal IS NOT NULL
            ) AS num_journals,
            (
                SELECT COUNT(*)
                FROM analysis{0}_fast_paper_authors AS pa
                WHERE
                pa.author_id = aa.author_id AND
                pa.date_created < %(cutoff_date)s AND
                pa.journal IN
                    ('nature', 'science', 'pnatlacadsciusa', 'physrevlett')
            ) AS num_papers_famous

            FROM `analysis{0}_authors` AS aa

            INNER JOIN analysis{0}_nc_data AS nc
            ON aa.author_id = nc.author_id AND
            nc.predict_after_years = %(predict_after_years)s

            INNER JOIN analysis{0}_hindex_data AS hi
            ON aa.author_id = hi.author_id AND
            hi.predict_after_years = %(predict_after_years)s

            ORDER BY aa.author_id ASC
            """.format(self.suffix_cuts)
        
        header = [
            "hindex_cumulative",
            "sqrt_nc_before",
            "hindex_before",
            "years_since_first_paper",
            "num_journals",
            "num_papers_famous"
            ]
        
        import csv
        for years in range(1, self.predict_after_years+1):
            
            orig_predict_after_years = self.predict_after_years
            self.predict_after_years = years
            self.generate_hindex_data()
            self.generate_nc_data()
            
            # Acuna et al get the whole dataset, not just training or
            # validation! The glmnet then does crossvalidation on this
            # by itself.
            c.execute(sql, {
                'cutoff_date': self.cutoff_date,
                'predict_after_years': years
                })
            
            file = os.path.join(self.evaluate_dir, 'acuna',
                                'data-%s.csv' % years)
            writer = csv.writer(open(file, 'w'))
            # writer.writerow(header)
            writer.writerows(c.fetchall())
            
            self.predict_after_years = orig_predict_after_years
   
    # From [1]
    # [1]http://blog.samuelmh.com/2015/02/pagerank-sparse-matrices-python-ipython.html
    def __compute_pagerank(self, G, beta=0.85, epsilon=10**-4):
        '''
        Efficient computation of the PageRank values using a sparse adjacency
        matrix and the iterative power method.

        Parameters
        ----------
        G : boolean adjacency matrix. np.bool8
            If the element j,i is True, means that there is a link from i to j.
        beta: 1-teleportation probability.
        epsilon: stop condition. Minimum allowed amount of change in the
            PageRanks between iterations.

        Returns
        -------
        output : tuple
            PageRank array normalized top one.
            Number of iterations.

        '''
        
        # Test adjacency matrix is OK
        n, _ = G.shape
        assert(G.shape == (n, n))
        # Constants Speed-UP
        deg_out_beta = G.sum(axis=0).T/beta  # vector
        # Initialize
        ranks = np.ones((n, 1))/n  # vector
        time = 0
        flag = True
        while flag:
            time += 1
            # Ignore division by 0 on ranks/deg_out_beta
            with np.errstate(divide='ignore'):
                new_ranks = G.dot((ranks/deg_out_beta))  # vector
            # Leaked PageRank
            new_ranks += (1-new_ranks.sum())/n
            # Stop condition
            if np.linalg.norm(ranks-new_ranks, ord=1) <= epsilon:
                flag = False
            ranks = new_ranks
        return(ranks, time)
    
    def test_pagerank(self):
        
        # This test shows that artifitially increasing 'numpapers' does not go
        # well with the pagerank algorithm!
        # --> Introduced continuous_id field in authors table
        numpapers = 5
        numedges = 5
        data = {
            'citing_paper': [0, 1, 2, 3, 4],
            'cited_paper': [1, 2, 3, 4, 0]
            }
        
        from scipy import sparse
        csr = sparse.csr_matrix(
            ([True]*numedges,
                (data['citing_paper'],
                 data['cited_paper'])),
            shape=(numpapers, numpapers))
        
        (ranks, time) = self.__compute_pagerank(csr)
        print(ranks)
        
        numpapers = 8
        numedges = 5
        data = {
            'citing_paper': [0, 1, 2, 3, 4],
            'cited_paper': [1, 2, 3, 4, 0]
            }
        
        from scipy import sparse
        csr = sparse.csr_matrix(
            ([True]*numedges,
                (data['citing_paper'],
                 data['cited_paper'])),
            shape=(numpapers, numpapers))
        
        (ranks, time) = self.__compute_pagerank(csr)
        print(ranks)
        
        self.__generate_paper_citation_pageranks(max_date=date(2004, 1, 1))
        self.__generate_coauthor_pageranks(max_date=date(2004, 1, 1))
    
    def __generate_paper_citation_pageranks(self, max_date):
        c = db().cursor()
        
        # Maybe: Restrict to papers <= max_date? If yes, need to take into
        # account that ids are not in the same order as date_created!
        c.execute("""SELECT MIN(id),MAX(id),COUNT(*) FROM papers""")
        for row in c:
            min_paper_id = row[0]
            max_paper_id = row[1]
            numpapers = row[2]
        
        if numpapers != max_paper_id - min_paper_id + 1:
            raise Exception("Ids are non-continuous?!")
        
        print("Loading data from DB...")
        numedges = c.execute("""
            SELECT c.citing_paper, c.cited_paper
            FROM citations AS c
            INNER JOIN papers AS p
            ON p.id = c.citing_paper AND p.num_authors < %(max_authors)s
            WHERE c.citing_paper != c.cited_paper
            AND p.date_created <= %(max_date)s
            """, {
                'max_authors': self.max_authors,
                'max_date': max_date})
        
        data = np.fromiter(
            c.fetchall(), count=numedges,
            dtype=[('citing_paper', 'i4'), ('cited_paper', 'i4')])
        
        print("Puttin data into csr matrix...")
        from scipy import sparse
        csr = sparse.csr_matrix(
            ([True]*numedges,
                (data['citing_paper']-min_paper_id,
                 data['cited_paper']-min_paper_id)),
            shape=(numpapers, numpapers))
        
        print("Calculating pagerank...")
        (ranks, time) = self.__compute_pagerank(csr)
        
        return (ranks, min_paper_id)
    
    def generate_paper_citation_pageranks(self):
        
        print("Generating paper citation pageranks...")
        
        # Calculate PageRank for each paper from citation graph
        
        # For perauthor cutoff:
        # In principle, should take into account for each author
        # papers up to first_paper_date+self.split_after_years
        # In practice, this takes too long. Instead, calculate
        # PageRanks for each step_months interval from
        # self.first_paper_min_date to
        # self.first_paper_max_date
        # and use the corresponding PageRank for each author
        
        # For single cutoff:
        # Just calculate pageranks once at cutoff date
        
        step_months = 6
        
        c = db().cursor()
        c2 = db().cursor()
        
        date = self.first_paper_max_date
        while date > self.first_paper_min_date:
            
            if self.cutoff == self.CUTOFF_PERAUTHOR:
                max_first_paper_date = date
                min_first_paper_date = date - relativedelta(months=step_months)
                max_date = self.get_split_date(date)
            elif self.cutoff == self.CUTOFF_SINGLE:
                # Need only one iteration
                max_first_paper_date = self.first_paper_max_date
                min_first_paper_date = self.first_paper_min_date
                max_date = self.get_split_date(date)
                date = self.first_paper_min_date
            else:
                raise Exception("Not implemented")
            
            print(date, max_date)
            
            # Calculate page ranks
            (ranks, id_offset) = self.__generate_paper_citation_pageranks(
                max_date=max_date)
            
            # Select papers whose first_paper_date is in
            # [date, date+step_months)
            print("Selecting papers")
            sql = """
                SELECT pa.paper_id, pa.author_id
                
                FROM analysis{0}_fast_paper_authors AS pa
                
                INNER JOIN analysis{0}_authors AS a
                ON a.author_id = pa.author_id
                    AND a.first_paper_date < %(max_first_paper_date)s
                    AND a.first_paper_date >= %(min_first_paper_date)s
                
                WHERE pa.date_created <= %(max_date)s
                """.format(self.suffix_cuts)
            numpapers = c.execute(sql, {
                'max_first_paper_date': max_first_paper_date,
                'min_first_paper_date': min_first_paper_date,
                'max_date': max_date})
            
            for row in c:
                paper_id = row[0]
                author_id = row[1]
                
                pagerank = ranks[paper_id - id_offset].item()
                
                print("-> ", paper_id, pagerank)
                sql = """
                    UPDATE analysis{0}_fast_paper_authors
                    SET pagerank = %(pagerank)s
                    WHERE paper_id = %(paper_id)s AND author_id = %(author_id)s
                    """.format(self.suffix_cuts)
                c2.execute(sql, {
                    'pagerank': pagerank,
                    'paper_id': paper_id,
                    'author_id': author_id})
            
            date = min_first_paper_date
            db().commit()
        
        print("Done")

    def __generate_coauthor_pageranks(self, max_date):
        c = db().cursor()
        
        c.execute("""SELECT
            MIN(continuous_id),
            MAX(continuous_id),
            COUNT(*)
            FROM authors""")
        for row in c:
            min_author_id = row[0]
            max_author_id = row[1]
            numauthors = row[2]
        
        if numauthors != max_author_id - min_author_id + 1:
            raise Exception("Ids are non-continuous?!")
        
        print("Loading data from DB...")
        numedges = c.execute("""
            SELECT a1.continuous_id, a2.continuous_id
            
            FROM paper_authors AS pa1
            INNER JOIN authors AS a1
            ON pa1.author_id = a1.id
            
            INNER JOIN paper_authors AS pa2
            ON pa2.paper_id = pa1.paper_id
                AND pa2.author_id != pa1.author_id
            INNER JOIN authors AS a2
            ON pa2.author_id = a2.id
            
            INNER JOIN papers AS p
            ON pa1.paper_id = p.id
                AND p.date_created <= %(max_date)s
                AND p.num_authors < %(max_authors)s
            """, {
                'max_authors': self.max_authors,
                'max_date': max_date})
        
        data = np.fromiter(
            c.fetchall(), count=numedges,
            dtype=[('author_id1', 'i4'), ('author_id2', 'i4')])
        
        print("Puttin data into csr matrix...")
        from scipy import sparse
        csr = sparse.csr_matrix(
            ([True]*numedges,
                (data['author_id1']-min_author_id,
                 data['author_id2']-min_author_id)),
            shape=(numauthors, numauthors))
        
        print("Calculating pagerank...")
        (ranks, time) = self.__compute_pagerank(csr)
        
        return (ranks, min_author_id)
     
    def generate_coauthor_pageranks(self):
        
        print("Generating coauthor pageranks")
        
        # Calculate PageRank for each author from coauthor graph
        
        # For perauthor cutoff:
        # In principle, should take into account for each author
        # papers up to first_paper_date+self.split_after_years
        # In practice, this takes too long. Instead, calculate
        # PageRanks for each step_months interval from
        # self.first_paper_min_date to
        # self.first_paper_max_date
        # and use the corresponding PageRank for each author
        
        # For single cutoff:
        # Just calculate pageranks once at cutoff date
        step_months = 6
        
        c = db().cursor()
        c2 = db().cursor()
        
        c.execute("""
            SELECT id, continuous_id
            FROM authors""")
        to_continous_id = {row[0]: row[1] for row in c}
        
        date = self.first_paper_max_date
        while date > self.first_paper_min_date:
            if self.cutoff == self.CUTOFF_PERAUTHOR:
                max_first_paper_date = date
                min_first_paper_date = date - relativedelta(months=step_months)
                max_date = self.get_split_date(date)
            elif self.cutoff == self.CUTOFF_SINGLE:
                # Need only one iteration
                max_first_paper_date = self.first_paper_max_date
                min_first_paper_date = self.first_paper_min_date
                max_date = self.get_split_date(date)
                date = self.first_paper_min_date
            else:
                raise Exception("Not implemented")
            
            print(date, max_date)
            
            # Calculate page ranks
            (ranks, id_offset) = self.__generate_coauthor_pageranks(
                max_date=max_date)
            
            # Select papers whose first_paper_date is in
            # [date, date+step_months)
            print("Selecting authors")
            sql = """
                SELECT a.author_id, coa.coauthor_id
                
                FROM analysis{0}_authors AS a
                
                INNER JOIN analysis{0}_fast_coauthors AS coa
                ON coa.analysis_author_id = a.author_id
                    AND coa.first_date <= %(max_date)s
                
                WHERE
                a.first_paper_date < %(max_first_paper_date)s AND
                a.first_paper_date >= %(min_first_paper_date)s
                """.format(self.suffix_cuts)
            num = c.execute(sql, {
                'max_date': max_date,
                'max_first_paper_date': max_first_paper_date,
                'min_first_paper_date': min_first_paper_date})
            
            for i, row in enumerate(c):
                analysis_author_id = row[0]
                coauthor_id = row[1]
                
                continous_coauthor_id = to_continous_id[coauthor_id]
                pagerank = ranks[continous_coauthor_id - id_offset].item()
                
                if i % 1000 == 0:
                    print("-> ", i, "/", num, ":", coauthor_id, pagerank)
                sql = """
                    UPDATE analysis{0}_fast_coauthors
                    SET coauthor_pagerank = %(pagerank)s
                    WHERE analysis_author_id = %(analysis_author_id)s
                    AND coauthor_id = %(coauthor_id)s
                    """.format(self.suffix_cuts)
                c2.execute(sql, {
                    'pagerank': pagerank,
                    'analysis_author_id': analysis_author_id,
                    'coauthor_id': coauthor_id})
            
            date = min_first_paper_date
            db().commit()
        
        print("Done")


if __name__ == '__main__':
    a = Analysis(cutoff=Analysis.CUTOFF_SINGLE)
    a.generate_authors()
    a.generate_fast_tables()
    orig_predict_after_years = a.predict_after_years
    for years in range(1, orig_predict_after_years+1):
        a.predict_after_years = years
        a.generate_hindex_data()
        a.generate_nc_data()
        a.predict_after_years = orig_predict_after_years
    a.plot_hirsch()
    a.generate_paper_citation_pageranks()
    a.generate_coauthor_pageranks()
