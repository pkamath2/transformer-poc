# Transformer - Raw Audio

This proof of concept is heavily inspired from Peter Bloem's amazing blogpost and videos at - http://peterbloem.nl/blog/transformers


## Training

See the notebook train.ipynb. Download NSynth dataset from https://magenta.tensorflow.org/datasets/nsynth#files (json/wav format) and update the `base_data_dir` and `labels_dir` with the local directory location.

I trained 2 models - one with 32 mins of data and another with 6 hours of data. Currently the code is configured to train on 6 hours of acoustic guitar.

## Sampling

See notebook called sample-with-*.ipynb for sampling. Both notebooks are the same, just created two to compare samples from different models/checkpoints. Update the checkpoint you want to use.
