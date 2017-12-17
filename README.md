# neural_nlp

Deep Neural Networks for natural language processing.

## Project structure
This repository is structured to have the following features:
 - [datagen](neural_nlp/datagen): handles data download and data iteration
 - models: generic model structures; not every hyperparameter is made generic, but these should be easily adjustable to
   your needs
 - [integrations](neural_nlp/integrations): specific model integration of a specific dataset
 - neural network utilities: TODO explain utilities

## [Datasets](neural_nlp/datagen)
These utility classes seamlessly download the required data and stores a local cache on your system. The data is
pre-processed and allows batch iteration over the dataset for use in neural networks.

### [Reuters-21578](neural_nlp/datagen)
These SML are parsed into json records, one per line. The fields currently exposed are title, text, topics, and mode.
The types of splits available for this dataset are modapte, lewissplit, cgisplit, and no split. See the README.txt
in the download directory for more details. The value of the mode is either `train` or `test`, depending on the split
value. Topics may have zero or more values.

### [Stackoverflow titles](neural_nlp/datagen)
This data comes from the 2015NAACL VSM-NLP workshop-"Short Text Clustering via Convolutional Neural Networks".
Please see http://naacl15vs.github.io/index.html and https://github.com/jacoxu/StackOverflow for details.

This class allows for multiple types of iteration over the data.
 - straight iteration as a label, title tuple
 - iteration of a "query," positive sample, and a set of negative examples; this is for input into the DSSM model
   described below
 - a batched version of the previous iteration

## [Models](neural_nlp/models)
### Deep Structured Semantic Model
See xxx for details.

## [Integrations](neural_nlp/integrations)
### stackoverflow dssm
This integration allows for a similarity model of stackoverflow titles. Here the "query" is a title. The positive
sample is a title from the same class and the negative samples are titles from different classes then the query class.

