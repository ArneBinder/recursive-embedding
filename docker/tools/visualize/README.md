# Visualize Corpus Data

TODO: add description

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

Adapt parameters in [`docker/create-corpus/sick/.env`](.env) file.

To start the REST endpoint, execute from project root:

```bash
cd docker/tools/visualize && docker-compose up
```

Optional:
 * install [Postman](https://www.getpostman.com/)
 * load [this Postman Collection](Visualize.postman_collection.json) into Postman



