FROM tensorflowfold_conda_tf1.3_mkl:latest

#RUN git clone https://github.com/ArneBinder/recursive-embedding.git && cd recursive-embedding

COPY requirements.txt recursive-embedding/requirements.txt
RUN pip --no-cache-dir install -r recursive-embedding/requirements.txt && python -m spacy download en_core_web_md
COPY src recursive-embedding/src
#VOLUME /root/corpora_in /root/corpora_out
WORKDIR /root/recursive-embedding/src