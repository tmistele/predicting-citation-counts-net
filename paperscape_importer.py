import settings
import os
from urllib.request import urlretrieve
from db import db


class PaperscapeImporter:
    
    year_first = 1991
    year_last = 2017
    
    base_url = 'https://github.com/paperscape/paperscape-data/raw/master/'
    
    data_dir = os.path.join(settings.DATA_DIR, 'paperscape')
    
    def __init__(self):
        # Create dir for raw files if it does not exist
        os.makedirs(self.data_dir, exist_ok=True)
    
    def __download_data(self):
        for year in range(self.year_first, self.year_last+1):
            file = os.path.join(self.data_dir, str(year) + '.csv')
            if not os.path.exists(file):
                url = self.base_url + 'pscp-' + str(year) + '.csv'
                print(url, file)
                urlretrieve(url, file)
            else:
                print("Already downloaded", year)
    
    def __import_to_db(self):
        
        c = db().cursor()
        c.execute("SELECT id, arxiv_id FROM papers")
        paper_ids = {}
        for row in c:
            paper_ids[row[1]] = row[0]
        
        for year in range(self.year_first, self.year_last+1):
            file = os.path.join(self.data_dir, str(year) + '.csv')
            for line in open(file, 'rt'):
                if line[0] == '#':
                    continue
                
                # Format is
                # arxiv-id;comma-separated-arxiv-categories;num-found-refs;num-total-refs;comma-separated-refs;comma-separated-authors;title
                
                data = line.split(";", 7)
                if data[2] == '0':
                    continue
                
                if data[0] not in paper_ids:
                    print("Skip", data[0])
                    continue
                
                citing_id = paper_ids[data[0]]
                print(data[0], citing_id)
                
                values = []
                for cited_paper in data[4].split(','):
                    if cited_paper not in paper_ids:
                        continue
                    values.append(
                        '(%s, %s)' % (citing_id, paper_ids[cited_paper])
                    )
                
                if len(values):
                    c.execute(
                        "INSERT INTO citations (citing_paper, cited_paper) " +
                        "VALUES " + (', '.join(values)))
            db().commit()
    
    def run(self):
        self.__download_data()
        self.__import_to_db()


if __name__ == '__main__':
    print("Running paperscape import")
    importer = PaperscapeImporter()
    importer.run()
