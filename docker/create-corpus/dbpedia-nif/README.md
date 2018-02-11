# Create corpus from DBpedia-NIF data

The [2016-10 release of DBpedia](http://wiki.dbpedia.org/datasets/dbpedia-version-2016-10) contains RDF [NIF](https://site.nlp2rdf.org/) data (see section 5. NLP Datasets of [Downloads 2016-10](http://wiki.dbpedia.org/downloads-2016-10)) covering the following: 
 * nif-context.ttl: The full text of a wiki page as the context for all subsequent information about this page.
 * nif-page-structure​.ttl: The structure of the wiki page as nif:Structure instances, such as Section, Paragraph and Title.
 * nif-text-links.ttl: All in-text links of a wiki page as nif:Word or nif:Phrase.

This docker-compose service converts this triple data into a simple tree serialization optimized for hierarchical neural 
network training. The corpus consists of:
 * a lexicon: it holds id-string mappings, id-stringhash mappings and embedding vectors
 * a forest: a serialized forest consists of two numpy arrays. One **data** array is holding the leaf ids or string hashes (can 
 be switched when a lexicon is provided), the other holds **parent** offsets for every data point. For training, the parent array 
 can be converted into two arrays, **children** and **children position**. The children array holds amounts of children for 
 every data point followed by the child offsets. The children position array indicates for every data point, where to look 
 in the children array or holds a negative value, if the data point has no children.

The workflow is as follows. Each article is split into its terminal NIF structure objects: nif:Paragraph or nif:Title. The terminals are parsed and serialized into trees by using the dependency structure. Thereby, artificial link nodes are appended to word nodes, that are contained in hyperlinks to other DBpedia resources. If a words parent links to the same resource, the link node is append to the parent only. Then, terminal trees are combined in means of the NIF structure elements that they contain (nif:Section). Note that nif:Structure elements can embed several nif:Structure elements again.
In its current state, the service uses only the first section of every article. To query the respective data, it is loaded into a local Virtuoso triple store.


## HOW TO build the corpus

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

Configure parameters in [`docker/create-corpus/dbpedia-nif/.env`](.env), especially set `HOST_VIRTUOSO_DATA` to the value of `db_dir` and `HOST_CORPORA_OUT` to the directory you want to output the corpus files (size: ~30 GB).

To start the processing, execute from repository root:

```bash
cd docker/create-corpus/dbpedia-nif && docker-compose up corpus-dbpedia-nif
```

NOTE: The processing can be interrupted any time, restarting continues from the latest position. (TODO: Verify that order of triples in main loop does not change!)


## TODO:
 * forest (batches): generate **children arrays** from parent DONE
 * forest (batches): calc **resource_offsets**: get root_offsets -> add 1 (positions of resource_ids) -> get data (resource_id) -> generate mapping {resource_id: root_offset} -> save as numpy
 * forest (batches): **count values**:  `np.unique(data, return_counts=True)` -> `unique.dump` and `count.dump`
 * lexicon: **merge**
 * counts: **merge**
 * lexicon & counts & forest (batches): set low frequency **words to UNKNOWN** (differentiate between resource_ids and other data)
 * lexicon: add **embedding vectors** to lexicon (from spacy nlp) -> *freezes* the lexicon (no more addition of entries, merging, etc. possible)
 * lexicon & forest (batches): convert string hashes to ids (would also *freeze* the lexicon)