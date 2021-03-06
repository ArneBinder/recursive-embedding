# Create a rec-emb data model from DBpedia-NIF

The [2016-10 release of DBpedia](http://wiki.dbpedia.org/datasets/dbpedia-version-2016-10) contains RDF [NIF](https://site.nlp2rdf.org/) data (see section 5. NLP Datasets of [Downloads 2016-10](http://wiki.dbpedia.org/downloads-2016-10)) covering the following: 
 * nif-context.ttl: The full text of a wiki page as the context for all subsequent information about this page.
 * nif-page-structure​.ttl: The structure of the wiki page as nif:Structure instances, such as Section, Paragraph and Title.
 * nif-text-links.ttl: All in-text links of a wiki page as nif:Word or nif:Phrase.

This docker-compose service converts this triple data into a simple tree serialization optimized for hierarchical neural 
network training (see [main readme](../../../README.md)).

The workflow is as follows. Each article is split into its terminal NIF structure objects: nif:Paragraph\[s\] and nif:Title\[s\]. The terminals are parsed and converted into trees by using its dependency structure. Thereby, an artificial link node is appended to every word node, that is contained in a hyperlink to another DBpedia resource. If a words parent links to the same resource, the link node is appended to the parent only. Then, terminal trees are combined in means of the NIF structure elements that they contain (nif:Section). Note that nif:Section elements can embed several nif:Section elements again.
In its current state, the service uses only the first section of every article, that corresponds to the article abstract, if it is available. To query the respective data, it is loaded into a local Virtuoso triple store.


## HOW TO build the data model

ATTENTION: The Virtuoso database files (in location `db_dir`) grow to a substantial size, i.e. ~100 GB.

Install:
 * docker
 * docker compose

The following steps rest on the detailed [guide by Jörn Hees](https://joernhees.de/blog/2015/11/23/setting-up-a-linked-data-mirror-from-rdf-dumps-dbpedia-2015-04-freebase-wikidata-linkedgeodata-with-virtuoso-7-2-1-and-docker-optional/).

Set up respective in-/out directories:
```bash
dump_dir=~/dumps/dbpedia-nif/2016-10
db_dir=~/virtuoso_db
mkdir -p "$dump_dir"
```
  
Download and prepare nif data (tested for EN, other languages should work, too), i.e.

download ontology files ([DBpedia Ontology](http://downloads.dbpedia.org/2016-10/dbpedia_2016-10.owl) and the [NIF core ontology](http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core/nif-core.owl)):
```bash
cd "$dump_dir"
mkdir classes
mkdir nif-core
cd classes
wget http://downloads.dbpedia.org/2015-04/dbpedia_2015-04.owl
cd ../nif-core 
wget http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core/nif-core.owl
```
 
and download triple files: [NIF Context](http://downloads.dbpedia.org/2016-10/core-i18n/en/nif_context_en.ttl.bz2), [NIF page structure](http://downloads.dbpedia.org/2016-10/core-i18n/en/nif_page_structure_en.ttl.bz2), [NIF Text Links](http://downloads.dbpedia.org/2016-10/core-i18n/en/nif_text_links_en.ttl.bz2)
```bash
cd "$dump_dir"
mkdir en
cd en
wget http://downloads.dbpedia.org/2016-10/core-i18n/en/nif_context_en.ttl.bz2
wget http://downloads.dbpedia.org/2016-10/core-i18n/en/nif_page_structure_en.ttl.bz2
wget http://downloads.dbpedia.org/2016-10/core-i18n/en/nif_text_links_en.ttl.bz2
```

Repack (Virtuoso does not accept .bz2 files):
```bash
cd "$dump_dir"/en
apt-get install pigz pbzip2
for i in */*.nt.bz2 ; do echo $i ; pbzip2 -dc "$i" | pigz - > "${i%bz2}gz" && rm "$i"; done
```

Prepare the virtuoso store:

```bash
docker run -d --name dbpedia-vadinst \
-v "$db_dir":/var/lib/virtuoso-opensource-7 \
joernhees/virtuoso run &&
docker exec dbpedia-vadinst wait_ready &&
docker exec dbpedia-vadinst isql-vt PROMPT=OFF VERBOSE=OFF BANNER=OFF \
"EXEC=vad_install('/usr/share/virtuoso-opensource-7/vad/rdf_mappers_dav.vad');" &&
docker exec dbpedia-vadinst isql-vt PROMPT=OFF VERBOSE=OFF BANNER=OFF \
"EXEC=vad_install('/usr/share/virtuoso-opensource-7/vad/dbpedia_dav.vad');" &&
docker stop dbpedia-vadinst &&
docker rm -v dbpedia-vadinst
```

Import classes into virtuoso store:
```bash
# dbpedia classes
docker run --rm \
    -v "$db_dir":/var/lib/virtuoso-opensource-7 \
    -v "$dump_dir"/classes:/import:ro \
    joernhees/virtuoso import 'http://dbpedia.org/resource/classes#'
# nif classes
docker run --rm \
    -v "$db_dir":/var/lib/virtuoso-opensource-7 \
    -v "$dump_dir"/nif-core:/import:ro \
    joernhees/virtuoso import 'http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#'
```

Import the actual data (will use 64 GB RAM and take about 1 hour)
```bash
docker run --rm \
    -v "$db_dir":/var/lib/virtuoso-opensource-7 \
    -v "$dump_dir"/en:/import:ro \
    -e "NumberOfBuffers=$((64*85000))" \
    joernhees/virtuoso import 'http://dbpedia.org/nif'
```

To test if everything went well, check out the Virtuoso Conductor web interface at [http://localhost:8891/conductor](http://localhost:8891/conductor (user: dba, pw: dba)) (with 10 GB RAM) after loading the Virtuoso docker image with the following command (quit ):
```bash
docker run --name dbpedia \
    -v "$db_dir":/var/lib/virtuoso-opensource-7 \
    -p 8891:8890 \
    -e "NumberOfBuffers=$((10*85000))" \
    joernhees/virtuoso run
```

Clone this repository:
```bash
git clone https://github.com/ArneBinder/recursive-embedding.git
cd recursive-embedding
```

Rename [`docker/create-corpus/dbpedia-nif/.env.dev`](.env.dev) (or copy) to `.env` and configure its parameters, especially set `HOST_VIRTUOSO_DATA` to the value of `db_dir` and `HOST_CORPORA_OUT` to the directory you want to output the corpus files (ATTENTION: expected corpus size: ~32 GB (batches) + X GB (final)).

To start the (batched) processing, execute from repository root:

```bash
cd docker/create-corpus/dbpedia-nif && docker-compose up corpus-parse
```
NOTE: The processing can be interrupted any time, restarting continues from the latest position.

This previous command creates batches of processed data. To merge the individual lexica into a final one, convert the string hashes to lexicon indices and finally concatenate all data, execute:
```bash
cd docker/create-corpus/dbpedia-nif && docker-compose up corpus-merge
```

Finally, we create (filtered, split and shuffled) index files with:
```bash
cd docker/create-corpus/dbpedia-nif && docker-compose up corpus-indices
```
The created files (extension: .idx.[id].npy) contain just the shuffled positions of the roots in the merged corpus. But they roots are filtered according to the number of see also references in this article/tree (see parameters SEEALSO_MIN_COUNT and SEEALSO_MAX_COUNT in the .env file for the actual values).


## TODO:
 * DONE forest (batches): generate **children arrays** from parent
 * DONE forest (batches): calc **resource_offsets**: get root_offsets -> dump (evtl. check against resource_hashes array: add 1 to root_offsets (gets positions of resource_ids) -> get data (resource_id hashes))
 * DONE forest (batches): **count values**:  `np.unique(data, return_counts=True)` -> `unique.dump` and `count.dump`
 * DONE lexicon: **merge**
 * DONE counts: **merge**
 * DONE lexicon & counts & forest (batches): set low frequency **words to UNKNOWN** (differentiate between resource_ids and other data)
 * DONE lexicon: add **embedding vectors** to lexicon (from spacy nlp) -> *freezes* the lexicon (no more addition of entries, merging, etc. possible)
 * DONE lexicon & forest (batches): convert string hashes to ids (would also *freeze* the lexicon)