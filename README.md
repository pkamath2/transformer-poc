# Transformer - Raw Audio

This proof of concept is heavily inspired from Peter Bloem's amazing blogpost and videos at - http://peterbloem.nl/blog/transformers


## Training

See the notebook train.ipynb. Download NSynth dataset from https://magenta.tensorflow.org/datasets/nsynth#files (json/wav format) and update the `base_data_dir` and `labels_dir` with the local directory location.

## Sampling

See notebook called sample.ipynb for sampling. Update the checkpoint you want to use.