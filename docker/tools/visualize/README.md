# Visualize Corpus Data

Visualize corpus data that is preprocessed and converted to rec-emb format.

Useful Postman collection: [Visualize](Visualize.postman_collection.json)

## HOW TO visualize

Install:
 * docker
 * docker compose

Clone this repo and switch into this folder:
```bash
git clone https://github.com/ArneBinder/recursive-embedding.git
cd recursive-embedding/docker/tools/visualize
```

Rename [`.env.dev`](.env.dev) (or copy) to `.env` and adapt its parameters.

To start the REST endpoint, execute from project root:

```bash
docker-compose up
```

Optional:
 * install [Postman](https://www.getpostman.com/)
 * load [this Postman Collection](Visualize.postman_collection.json) into Postman: File \> Import \> Import from Link
 * set up Postman environment variable:
    - `api_endpoint`=`http://0.0.0.0:5000/api`



