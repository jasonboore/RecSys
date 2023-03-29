# MKR


### Files in the folder

- `data/`
  - `book/`
    - `BX-Book-Ratings.csv`: raw rating file of Book-Crossing dataset;
    - `item_index2entity_id.txt`: the mapping from item indices in the raw rating file to entity IDs in the KG;
    - `kg.txt`: knowledge graph file;
  - `movie/`
    - `item_index2entity_id.txt`: the mapping from item indices in the raw rating file to entity IDs in the KG;
    - `kg.txt`: knowledge graph file;
    - `ratrings.dat`: raw rating file of MovieLens-1M;
  - `music/`
    - `item_index2entity_id.txt`: the mapping from item indices in the raw rating file to entity IDs in the KG;
    - `kg.txt`: knowledge graph file;
    - `user_artists.dat`: raw rating file of Last.FM;
- `src/`: implementations of MKR.




### Running the code
- Book
  - ```
    $ cd src
	$ python train.py --train-data ../data/book/rating_train.dat --test-data ../data/book/rating_test.dat --lam 0.025 --max-iter 100 --model-name dblp --rec 1 --large 2 --d 8 --vectors-u ../data/book/vectors_u.dat --vectors-v ../data/book/vectors_v.dat
    $ python preprocess.py --dataset book
    $ python main.py
    ```
    ```