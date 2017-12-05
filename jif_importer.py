import settings
import os
import csv
from db import db


class JifImporter:
    
    year_first = 2001
    year_last = 2009

    data_dir = os.path.join(settings.DATA_DIR, 'thomsonreuters')
    
    def __init__(self):
        pass
    
    def run(self):
        
        c = db().cursor()
        
        for year in range(self.year_first, self.year_last+1):
            file = os.path.join(self.data_dir,
                                'JournalHomeGrid-' + str(year) + '.csv')
            
            for row in csv.reader(open(file, 'r')):
                
                # Title row
                if row[0].startswith('Journal Data Filtered By:  '+
                                     'Selected JCR Year: '+str(year)):
                    continue
                # Header row
                if row[0] == 'Rank':
                    continue
                
                # Copyright row
                if row[0].startswith('Copyright') or \
                   row[0].startswith('By exporting'):
                    continue
                
                # Actual content
                journal = row[2]
                issn = row[3]
                jif = row[5]
                
                if jif == 'Not Available':
                    continue
                
                # Normalize journal
                journal = journal.lower() \
                    .replace(' ', '').replace(':', '').replace('-', '')
                
                print(journal, year, jif)
                
                c.execute("""
                    INSERT INTO jif SET
                    journal = %(journal)s,
                    issn = %(issn)s,
                    year = %(year)s,
                    jif = %(jif)s""", {
                        'journal': journal,
                        'issn': issn,
                        'year': year,
                        'jif': jif})
        
        print("Committing...")
        db().commit()


if __name__ == '__main__':
    importer = JifImporter()
    importer.run()
