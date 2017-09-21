## Requirements
```
pip install tensorflow matplotlib
```

Install [mantaflow](http://mantaflow.com/install.html). Run CMake with these options:
```
cmake .. -DGUI=ON -DOPENMP=ON -DNUMPY=ON
```

## Run

Commands are to be executed in the project root directory.

### Generate data

```
python generate_fluid_data.py
```

Simulation parameters can be adjusted in `utils.py`.

### Run training and test

```
python tf_super_network.py
```

### Visualize

```
python visualization.py test_output.npy
```
