# Self-Attention Implementation (ML)

This repository contains an implementation of self-attention mechanism for machine learning.

## About

Self-attention is a key component in transformer architectures, allowing models to weigh the importance of different parts of the input when processing sequences.

## Getting Started

This was a project to show the implematation of self attention in a sequence of words ."Dream big and work for it" was the sentense and its tokenized as words.Each words were considered as tokens.In self attention these tokens are the input embeddings with 3 features in it.so 6 tokens with 3 features in it makes it as a matrix of 6 X 3 (input embeddings).these vectors were plotted on the 3d graph to analyse the direction and magnitude of the vectors.

then we started calculting the attention score for each words.
Attention score is the dot product of vectors ,which is a scalar value.Attention score is calculated between the query and every single keys.Here I chose "Big" as the query and each tokens are the "Keys".

then when we sum up the attention scores ,it will not be equal to 1 ,so we have to normalize these scores .For that we can do simple vanilla normalization and also softmax normalization.Here we chose Tensorflow's Softmax Normalization.After normalization we get the Attention weights which can sum up to get 1 .This shows the percentage of relationwith other vectors from one vector.

Normalization improves the inter operability and also maintain the LLM training stability.

To get the Context Vector , Input embeddings X attention weights gives the context Vectors.

Thus we get Context vectors using self attention.
