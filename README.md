# recursive-dependency-embedding

Train embeddings for syntactic dependency edges. This enables recursive calculation of sentence embeddings with regard to syntactic dependency parses.

## prerequisites

The preprocessing expects a comma separated csv file with two columns representing IDs and contents, where content can be any utf-8 decodable kind of text.
For an example corpus, have a look at [the data](https://storage.googleapis.com/lateral-datadumps/wikipedia_utf8_filtered_20pageviews.csv.gz) represented in the article [The Unknown Perils of Mining Wikipedia](https://blog.lateral.io/2015/06/the-unknown-perils-of-mining-wikipedia/).

## License

Copyright 2017 Arne Binder

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

