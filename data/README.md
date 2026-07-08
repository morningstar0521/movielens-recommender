# Data

Place the MovieLens 25M dataset here. Expected layout:

```
data/
└── ml-25m/
    ├── ratings.csv     # userId,movieId,rating,timestamp
    ├── movies.csv      # movieId,title,genres
    ├── links.csv
    ├── tags.csv
    └── README.txt
```

Download:

```bash
curl -L -o ml-25m.zip https://files.grouplens.org/datasets/movielens/ml-25m.zip
unzip ml-25m.zip -d data/
```

Notes:
- ratings.csv is ~650 MB. Loading it as float64 takes ~3 GB RAM. The
  loader downcasts dtypes so it fits in well under 1 GB.
- genome-scores.csv (~430 MB) is not used by this project; you can
  delete it to save disk.
