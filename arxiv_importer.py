import settings
import os
from urllib.request import urlretrieve
from urllib.error import HTTPError
import xml.etree.ElementTree as ET
from unidecode import unidecode
from time import sleep
from db import db
import pickle


class ArxivImporter:
    
    url_initial = 'http://export.arxiv.org/oai2' + \
        '?verb=ListRecords&set=%s&metadataPrefix=arXiv' % settings.ARXIV_SET
    url_template_resumed = 'http://export.arxiv.org/oai2' + \
        '?verb=ListRecords&resumptionToken=%s'
    
    ns = {
        'oai': 'http://www.openarchives.org/OAI/2.0/',
        'arxiv': 'http://arxiv.org/OAI/arXiv/',
    }
    
    data_dir = os.path.join(settings.DATA_DIR, 'arxiv')
    file_template = settings.ARXIV_SET + '_%s.xml'
    
    def __init__(self):
        
        self.__category_cache = {}
        
        # Create dir for raw files if it does not exist
        os.makedirs(self.data_dir, exist_ok=True)
    
    def __download_data(self):
        
        i = 0
        file = os.path.join(self.data_dir, self.file_template % i)
        if os.path.exists(file):
            print("Already downloaded")
            return
        
        print("Retrieving initial", self.url_initial, file)
        urlretrieve(self.url_initial, file)
        
        i += 1
        resumption_token_element = ET.parse(file).getroot() \
            .find('oai:ListRecords', self.ns) \
            .find('oai:resumptionToken', self.ns)
        
        # Avoid throttling
        sleep(5)
        
        while resumption_token_element.text is not None:
            file = os.path.join(self.data_dir, self.file_template % i)
            url = self.url_template_resumed % resumption_token_element.text
            
            print("Retrieving", url, file)
            try:
                urlretrieve(url, file)
            except HTTPError as e:
                if e.code != 503:
                    raise e
                seconds = int(e.headers.get('retry-after', 31))
                print("Retry after %s seconds" % seconds)
                sleep(seconds+1)
                continue
            
            i += 1
            resumption_token_element = ET.parse(file).getroot() \
                .find('oai:ListRecords', self.ns) \
                .find('oai:resumptionToken', self.ns)
            
            # Avoid throttling
            sleep(5)
    
    def __papers_generator(self):
        i = 0
        file = os.path.join(self.data_dir, self.file_template % i)
        while os.path.exists(file):
            
            print(file)
            
            for record_element in ET.parse(file).getroot() \
                .find('oai:ListRecords', self.ns) \
                .findall('oai:record', self.ns):
                    
                    metadata_element = record_element \
                        .find('oai:metadata', self.ns) \
                    
                    # no metadata for deleted records, e.g. 1008.3138
                    if metadata_element is not None:
                        yield metadata_element \
                            .find('arxiv:arXiv', self.ns)
            
            i += 1
            file = os.path.join(self.data_dir, self.file_template % i)
    
    def __create_or_fetch_category(self, category_name):
        c = db().cursor()
        c.execute(
            """SELECT id FROM categories WHERE name = %s""",
            (category_name,)
        )
        if c.rowcount:
            self.__category_cache[category_name] = c.fetchone()[0]
        else:
            c.execute(
                """INSERT INTO categories SET name = %s""",
                (category_name,)
            )
            self.__category_cache[category_name] = c.lastrowid
        
        return self.__category_cache[category_name]
    
    def __get_category_ids(self, category_names):
        return [
            self.__category_cache[x] if x in self.__category_cache
            else self.__create_or_fetch_category(x) for x in category_names]

    def __import_to_db(self):
        
        # Initialize author ids from backend
        print("Loading backend author data...")
        (backend_author_ids, _) = pickle.load(open(
            os.path.join(self.data_dir, 'keywords-backend', 'author_ids'), 'rb'
        ))
        
        c = db().cursor()
        c.execute("SELECT COUNT(*) FROM authors")
        if c.fetchone()[0] == 0:
            print("Creating backend authors in database")
            i = 0
            # TODO: author_id == 0 fails?
            for name, id in backend_author_ids.items():
                db().cursor().execute(
                    """REPLACE INTO authors SET id=%s, name=%s""",
                    (id, name)
                )
                i += 1
                if i % 10000 == 0:
                    print(i)
            db().commit()
        
        print("Importing papers...")
        i = 0
        for metadata in self.__papers_generator():
            
            # Paper itself
            arxiv_id = metadata.find('arxiv:id', self.ns).text
            print(arxiv_id)
            created = metadata.find('arxiv:created', self.ns).text
            updated = metadata.find('arxiv:updated', self.ns)
            if updated is not None:
                updated = updated.text
            
            c.execute(
                """INSERT INTO papers
                SET arxiv_id=%s, date_created=%s, date_updated=%s""",
                (arxiv_id, created, updated)
            )

            paper_id = c.lastrowid
            
            # Authors
            author_values = []
            for author in metadata.find('arxiv:authors', self.ns).getchildren():
                keyname = author.find('arxiv:keyname', self.ns).text
                forenames = author.find('arxiv:forenames', self.ns)
                forenames = forenames.text if forenames is not None else ''
                name = ' '.join(
                    (forenames + ' ' + keyname)
                    .replace(',', '').replace('.', '').lower().split()
                )
                name = unidecode(name)
                if name in backend_author_ids:
                    author_id = backend_author_ids[name]
                    author_value = '(%s, %s)' % (paper_id, author_id)
                    if author_value not in author_values:
                        author_values.append(author_value)
            
            if len(author_values):
                c.execute("INSERT INTO paper_authors (paper_id, author_id) " +
                          "VALUES " + (', '.join(author_values)))
            
            # Categories
            category_ids = self.__get_category_ids(
                metadata.find('arxiv:categories', self.ns).text.split()
            )
            
            if len(category_ids):
                category_ids = ['(%s, %s)' % (paper_id, x)
                                for x in category_ids]
                c.execute(
                    "INSERT INTO paper_categories (paper_id, category_id) " +
                    "VALUES " + (', '.join(category_ids)))
            
            i += 1
            if i % 1000 == 0:
                print("Committing")
                # Commit changes
                db().commit()
  
        # Commit changes
        db().commit()
    
    def __add_paper_lengths(self):
        print("Adding paper lengths...")
        import json
        paper_lengths = json.load(open(os.path.join(
            self.data_dir, 'keywords-backend', 'all_lengths.json'), 'r'
        ))
        
        c = db().cursor()
        
        import re
        pattern = re.compile("\d")
        
        for paper_id, length in paper_lengths.items():
            
            # Restore '/'. e.g. hep-th34523 -> hep-th/34523
            pos = pattern.search(paper_id).start()
            if pos > 0:
                paper_id = paper_id[:pos] + '/' + paper_id[pos:]
            
            print(paper_id, length)
            c.execute("UPDATE papers SET length=%(length)s " +
                      "WHERE arxiv_id = %(arxiv_id)s", {
                        'length': length,
                        'arxiv_id': paper_id})
        
        print("Committing")
        db().commit()
    
    def __add_author_broadness(self):
        print("Adding broadness...")
        backend_broadness = pickle.load(open(
            os.path.join(self.data_dir, 'keywords-backend', 'broadness'),
            'rb'
        ))
        
        c = db().cursor()
        for author_id, broadness in enumerate(backend_broadness):
            print(author_id, broadness)
            c.execute("UPDATE authors SET broadness=%(broadness)s " +
                      "WHERE id = %(author_id)s", {
                        'broadness': broadness,
                        'author_id': author_id})
        
        print("Committing")
        db().commit()
    
    def __add_author_broadness_lda(self):
        print("Adding broadness_lda...")
        backend_broadness_lda = pickle.load(open(
            os.path.join(self.data_dir, 'keywords-backend', 'lda_broadness'),
            'rb'
        ))
        
        c = db().cursor()
        for author_id, broadness_lda in enumerate(backend_broadness_lda):
            print(author_id, broadness_lda)
            c.execute("UPDATE authors SET broadness_lda=%(broadness_lda)s " +
                      "WHERE id = %(author_id)s", {
                        'broadness_lda': broadness_lda,
                        'author_id': author_id})
        
        print("Committing")
        db().commit()
    
    def __add_jifs(self):
        import re
        pattern = re.compile("\d")
        
        c = db().cursor()
        
        """
        SELECT papers.journal, arxiv_id, COUNT(*)
        FROM `papers`
        LEFT JOIN jif
        ON jif.journal = papers.journal
        WHERE papers.journal IS NOT NULL AND jif.journal IS NULL AND papers.journal != ''
        GROUP BY papers.journal HAVING COUNT(*) > 10
        ORDER BY COUNT(*) DESC
        """
        
        map = {
            'jhep': 'jhighenergyphys',
            'journalofhighenergyphysics': 'jhighenergyphys',
            'monnotroyastronsoc': 'monnotrastronsoc',
            'mnras': 'monnotrastronsoc',
            'monthlynoticesoftheroyalastronomicalsociety': 'monnotrastronsoc',
            'journalofappliedphysics': 'japplphys',
            'jquantspectroscradiattransf': 'jquantspectroscra',
            'jquantspectroscradiattransfer': 'jquantspectroscra',
            'classquantgrav': 'classicalquantgrav',
            'classquantumgrav': 'classicalquantgrav',
            'physicalreviewb': 'physrevb',
            'prb': 'physrevb',
            'physicalreviewa': 'physreva',
            'physicalreviewe': 'physreve',
            'physicalreviewc': 'physrevc',
            'physicalreviewd': 'physrevd',
            'physicalreviewletters': 'physrevlett',
            'physrevlettv': 'physrevlett',
            'prl': 'physrevlett',
            'europhyslett': 'epleurophyslett',
            'europhysicsletters': 'epleurophyslett',
            'epl': 'epleurophyslett',
            'a&a': 'astronastrophys',
            'astronomy&astrophysics': 'astronastrophys',
            'apj': 'astrophysj',
            'astrophysicaljournal': 'astrophysj',
            'theastrophysicaljournal': 'astrophysj',
            'nuclphysprocsuppl': 'nuclphysbprocsup',
            'jcap': 'jcosmolastropartp',
            'jphysa': 'jphysamathgen',
            'jphysg': 'jphysgnuclpartic',
            'jphysgnuclpartphys': 'jphysgnuclpartic',
            'jphyscondensmatter': 'jphyscondensmat',
            'genrelgrav': 'genrelatgravit',
            'jstatmech': 'jstatmechtheorye',
            'naturephysics': 'natphys',
            'physicslettersa': 'physletta',
            'physicslettersb': 'physlettb',
            'annalsphys': 'annphysnewyork',
            'annalsofphysics': 'annphysnewyork',
            'sigma': 'symmetryintegrgeom',
            'appliedphysicsletters': 'applphyslett',
            'physatomnucl': 'physatomnucl+',
            'opticsexpress': 'optexpress',
            'jetplett': 'jetplett+',
            'jetpletters': 'jetplett+',
            'astrophysjsuppl': 'astrophysjsuppls',
            'apjs': 'astrophysjsuppls',
            'theastrophysicaljournalsupplementseries': 'astrophysjsuppls',
            'fortschphys': 'fortschrphys',
            'opticsletters': 'optlett',
            'newjournalofphysics': 'newjphys',
            'jphysbatmoloptphys': 'jphysbatmolopt',
            'jphysb': 'jphysbatmolopt',
            'theormathphys': 'theormathphys+',
            'progtheorphyssuppl': 'progtheorphyssupp',
            'actaphyspolonb': 'actaphyspolb',
            'actaphyspolona': 'actaphyspola',
            'physrept': 'physrep',
            'astroparticlephysics': 'astropartphys',
            'astronomyandastrophysics': 'astronastrophys',
            'astronomy&astrophysics': 'astronastrophys',
            'nuovocimb': 'nuovocimentob',
            'nuclearphysicssectionb': 'nuclphysb',
            'publastronsocjap': 'publastronsocjpn',
            'supercondscitechnol': 'supercondscitech',
            'naturematerials': 'natmater',
            'jinst': 'jinstrum',
            'pnas': 'pnatlacadsciusa',
        }
        
        i = 0
        for metadata in self.__papers_generator():
            arxiv_id = metadata.find('arxiv:id', self.ns).text
            journal_ref = metadata.find('arxiv:journal-ref', self.ns)
            
            if journal_ref is None:
                continue
            journal_ref = journal_ref.text.lower() \
                .replace(' ', '').replace(':', '').replace('-', '') \
                .replace('.', '').replace(',', '').replace('(', '')
            
            pos = pattern.search(journal_ref)
            if pos is None:
                continue
            pos = pos.start()
            
            journal = journal_ref[0:pos]
            
            # Remove 'vol' at right hand side
            if journal.endswith('vol'):
                journal = journal[:-len('vol')]
            if journal.endswith('volume'):
                journal = journal[:-len('volume')]
            
            # Fixes
            if journal in map:
                journal = map[journal]
           
            print(arxiv_id, journal)
            c.execute("""UPDATE papers SET journal=%(journal)s
                      WHERE arxiv_id = %(arxiv_id)s""", {
                          'journal': journal,
                          'arxiv_id': arxiv_id})
            
            i += 1
            if i % 10000 == 0:
                print("Committing...")
                db().commit()
        
        db().commit()
        print("Done")
    
    def __add_paper_authors_countries(self):
        
        countries = pickle.load(open(os.path.join(
            self.data_dir, 'keywords-backend', 'author_paper_countries'), 'rb'
        ))
        
        c = db().cursor()
        
        import re
        pattern = re.compile("\d")
        
        for author_id, author_countries in enumerate(countries):
            print(author_id)
            for paper_id, country in author_countries:
                # Restore '/'. e.g. hep-th34523 -> hep-th/34523
                pos = pattern.search(paper_id).start()
                if pos > 0:
                    paper_id = paper_id[:pos] + '/' + paper_id[pos:]
                
                # Update
                c.execute("""UPDATE paper_authors AS pa
                    INNER JOIN papers AS p
                    ON p.id = pa.paper_id
                    SET pa.country = %(country)s
                    WHERE p.arxiv_id = %(arxiv_id)s AND
                    pa.author_id = %(author_id)s""", {
                        'country': country,
                        'author_id': author_id,
                        'arxiv_id': paper_id
                        })
        
        print("Committing...")
        db().commit()
    
    def __add_num_authors(self):
        db().cursor().execute("""UPDATE papers AS p
            SET p.num_authors = (
                SELECT COUNT(*)
                FROM paper_authors AS pa
                WHERE pa.paper_id = p.id
            )""")
        db().commit()
    
    def run(self):
        self.__download_data()
        self.__import_to_db()
        self.__add_paper_lengths()
        # Unused at the moment
        # self.__add_author_broadness()
        self.__add_author_broadness_lda()
        self.__add_jifs()
        self.__add_num_authors()
        # Unused at the moment
        # self.__add_paper_authors_countries()


if __name__ == '__main__':
    print("Running arXiv import")
    importer = ArxivImporter()
    importer.run()
    # importer.test()
