# Visualize Corpus Data

Visualize corpus data that is preprocessed and converted to rec-emb format.

Useful Postman collection: [Visualize](Visualize.postman_collection.json)

## HOW TO visualize

Install:
 * docker
 * docker compose

Clone this repo:
```bash
git clone https://github.com/ArneBinder/recursive-embedding.git
cd recursive-embedding
```

Rename [`docker/tools/visualize/.env.dev`](.env.dev) (or copy) to `.env` and adapt its parameters.

To start the REST endpoint, execute from project root:

```bash
cd docker/tools/visualize && docker-compose up
```

Optional:
 * install [Postman](https://www.getpostman.com/)
 * load [this Postman Collection](https://raw.githubusercontent.com/ArneBinder/recursive-embedding/master/docker/tools/visualize/Visualize.postman_collection.json) into Postman: File \> Import \> Import from Link



