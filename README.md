# Predicting Citation Counts with a Neural Network

This is the code used for [this paper](https://arxiv.org/abs/1806.04641).
Reproducing the results from this paper can be done as follows:

0. Download files. We plan to make these available on a webserver in the future. For now, you can ask us for these files and save the following ones in data/arxiv/keywords-backend/

```
papers
paper_topics
all_lengths.json
broadness_lda
```

and these in data/arxiv/thomsonreuters/

```
JournalHomeGrid-2001.csv
...
JournalHomeGrid-2009.csv
```

1. Set up a MySQL database and save the connection data in settings_private.py.
```
DB_PASS = '...'
DB_USER = '...'
DB_HOST = '...'
DB_NAME = '...'
```
2. Set up the database and import the arXiv/Paperscape/JIF data.

```bash
mysql < database_structure.sql
python arxiv_importer.py
python paperscape_importer.py
python jif_importer.py
```

3. Some pre-processing needs to be done.

```bash
python analysis.py
python net.py
```

4. Run the following SQL command.

```SQL
UPDATE analysissingle512_authors SET train_real = train
```

5. Generate the cross-validation groups and prepare the x and y data.

```bash
python run_local.py prepare
```

6. Train the neural network and random forest models for each cross-validation round $i (0 to 19).

```bash
python run_cluster.py train-rf $i
python run_cluster.py train-net $i
```

7. Evaluate the trained models as well as some naive baseline models for each $i and summarize the results.

```bash
python run_local.py evaluate-rf --i $i
python run_local.py evaluate-net --i $i
python run_local.py evaluate-linear-naive --i $i
python run_local.py summarize
```

The summary files will be placed in data/analysissingle512/evaluate/no-max-hindex, the results for each individual trained model will be placed in data/analysissingle512/evaluate/no-max-hindex/task-results.
