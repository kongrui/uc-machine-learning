# Machine Learning Engineer Nanodegree
## Capstone Proposal
July 21st, 2018

## Proposal



### Domain Background

https://www.kaggle.com/c/home-depot-product-search-relevance

My interested domain is the shopping/commerce search. I found this challenge from Kraggle.

Home Depot is asking Kagglers to help them improve their customers' shopping experience
by developing a model that can accurately predict the relevance of search results.


### Problem Statement

Search relevancy is an implicit measure Home Depot uses to gauge how quickly they can
get customers to the right products.

Currently, human raters evaluate the impact of potential changes to their search algorithms, which is a slow and subjective process.
Human rating is a slow and expensive process.
process. There are almost infinite search/product pairs so it is impossible for human raters to

By removing or minimizing human input in search relevance evaluation, Home Depot hopes to
increase the number of iterations their team can perform on the current search algorithms.
Historical information relevant to the project should be included.

Instead, this proposal is to build a model based on
search/product pair and its relevance score and use it to predict the relevance score of out-of-sample
pairs.

Per each query/product pair, human rater reads the product description and attributes, then gives the relevance score.
This is how data is produced.

This problem is clearly a regression problem.


### Datasets and Inputs

To create the ground truth labels, Home Depot has crowdsourced the search/product pairs to multiple human raters.

The relevance is a number between 1 (not relevant) to 3 (highly relevant).
Each pair was evaluated by at least three human raters. The provided relevance scores are the average value of the ratings.

There are three additional things to know about the ratings:
- The specific instructions given to the raters is provided in relevance_instructions.docx.
- Raters did not have access to the attributes.
- Raters had access to product images, while the competition does not include images.

Training Data - training.csv
- search_term
- product_title
- relevance

Knewledge data
- product_descriptions.csv : contains a text description of each product.
- attributes.csv : provides extended information about a subset of the products

### Solution Statement


### Benchmark Model


### Evaluation Metrics
_(approx. 1-2 paragraphs)_

In this section, propose at least one evaluation metric that can be used to quantify the performance of both the benchmark model and the solution model. The evaluation metric(s) you propose should be appropriate given the context of the data, the problem statement, and the intended solution. Describe how the evaluation metric(s) are derived and provide an example of their mathematical representations (if applicable). Complex evaluation metrics should be clearly defined and quantifiable (can be expressed in mathematical or logical terms).

### Project Design

A theoretical workflow for approaching a solution given the problem.
