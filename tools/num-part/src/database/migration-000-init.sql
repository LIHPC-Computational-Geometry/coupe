CREATE TABLE seed
    ( id INTEGER PRIMARY KEY
    , bytes BLOB NOT NULL
    , UNIQUE(bytes)
    );

CREATE TABLE distribution
    ( id INTEGER PRIMARY KEY
    , name TEXT NOT NULL
    , param1 REAL
    , param2 REAL
    , param3 REAL
    , UNIQUE(name, param1, param2, param3)
    );

CREATE TABLE experiment
    ( id INTEGER PRIMARY KEY
    , algorithm TEXT NOT NULL
    , seed INTEGER NOT NULL REFERENCES seed(id)
    , distribution INTEGER NOT NULL REFERENCES distribution(id)
    , sample_size INTEGER NOT NULL
    , iteration INTEGER NOT NULL
    , case_type INTEGER NOT NULL
    , imbalance REAL NOT NULL
    , algo_iterations INTEGER
    , UNIQUE(seed, iteration, distribution, sample_size, algorithm, case_type)
    );

CREATE INDEX experiment_distribution ON experiment (distribution);
