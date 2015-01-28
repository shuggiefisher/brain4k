# What is brain4k?


brain4k, pronounced "brainfork", is a framework that intends to improve the reproducability and
shareability of machine learning models.  Think of it as MVC for machine learning,
except with brain4k pipelines you have Data, Transforms, and Metrics.

The rules of brain4k:
1. A brain4k pipeline lives in version control, and so can be forked, reverted and managed like other code.
2. Each stage in the pipeline is deterministic and reproducible for a given commit
3. brain4k is framework and language agnostic - pipe from one language to another if your execution environment supports it
4. Every brain4k pipeline publishes performance metrics to encourage quality models to rise to the top
5. Contribute plugins for your preferred ML framework back to the community

## Sample pipelines

- [Extract image features from a convolutional neural network]()
- [Train a classifier on image features extracted from a convolutional neural network]()

## Installing brain4k

pip install

## Executing a pipeline

Clone one of our sample pipelines

```git clone repo-name local-path-to-repo```

ensuring you have installed the dependencies listed in the README

```brain4k local-path-to-repo```

## Publishing a pipeline

1. Push your repo somewhere public
2. If you want others to be able to reproduce your model, change any data blobs with a "local_filename" in favour of a "url" and sha1 hash for the file.
3. Tell us about it.

