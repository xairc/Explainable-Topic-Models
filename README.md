# Explainable Topic Models

## Overview
This project is generating explainable results of topic models. The results are not only topics but also other hidden variables.

Now, we open the source file to figure out the prototypical document from Aspect and Sentiment Unification Model (ASUM, Jo and Oh, 2011).

We will open other source files for various topic models.

## Requirements (testing environment)
- Python 3.6.3
- Numpy 1.14.3
- Scipy 0.19.1

## Structure of the source file
The source file name convention is as follow:

- Model name + Explainable method + Inference method.py
 - A moudle file to run the topic model with explainable outputs
 - i.e. `ASUMCaseGibbs.py` means finding prototypical document for the topic of ASUM, and infrence method of ASUM is Gibbs sampling.
 - Explainable method can be removed. It is the original topic model inference.

- Run_ + Model name + Explainable method + Inference method.py
 - An example file to run the topic model.
 - i.e. `Run_ASUMCaseGibbs.py` is example code to run `ASUMCaseGibbs.py`

## How to run
To run the program, you would be better to read running example file at first. It requires some inputs from command line arguments.

For example, to run the `Run_ASUMCaseGibbs.py`, five arguments are needed.
- document_file_path: the file path of document file
- voca_file_path = the file path of vocabulary file
- senti_words_prefix_path = the file path of predefined sentiment words for weighting each sentiment
- output_file_name = the prefix name of output files
- topics = the number of topics

## Supports
Please write questions in this repos issues section if you meet any problems.

## Reference
- Jo, Yohan, and Alice H. Oh. "Aspect and sentiment unification model for online review analysis." Proceedings of the fourth ACM international conference on Web search and data mining. ACM, 2011.
