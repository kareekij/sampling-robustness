# Sampling Robustness
Measuring Sampling Robustness of Complex Networks


### Generate samples
```
python sample_mising_data.py <network_file> -budget <budget> -experiment <no_of_experiment>
```

e.g.
```
python sample_missing_data.py ./example/socfb-Amherst41.mtxt
```

The generated samples are put in `./samples/` folder. It generates samples with different values of error probability `p` (between `0<p<0.5`).

By default, it generates 10 samples for each `p`. (You can use `-experiment <no_of_sample>` to define the number of output samples.)

### Measure Robustness

Execute this file to calculate the sampling robustness of the samples.
```
python calculate_robustness_from_sample.py <path_to_file>
```

e.g.
```
python calculate_robustness_from_sample.py ./data/grad_edges.txt
```

The output `output_robustness.txt` is stored in `log` folder (by default).


`adversary_sample_robustness_norm.py`

### Predicting Sampling Robustness
1. `./log/sample-training-aggregate.txt`
2. `./log/sample-testing-aggregate.txt`
```
python adversary_predict_r.py
```