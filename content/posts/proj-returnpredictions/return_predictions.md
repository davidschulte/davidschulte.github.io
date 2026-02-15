Oj<a href="../index.html" class="logo">Project</a>

- [Portfolio](../index.html)
- [About me](../aboutme.html)

<!-- -->

- <a href="https://github.com/davidschulte"
  class="icon brands fa-github"><span class="label">GitHub</span></a>
- <a href="https://linkedin.com/in/davidsiriusschulte"
  class="icon brands fa-linkedin"><span class="label">Linkedin</span></a>

<span class="date">February 2021</span>

# Customer return predictions

Given the order of a customer in an online shop, can we predict which
items will be returned later?

- <a href="https://github.com/davidschulte/bads2021"
  class="icon brands alt fa-github"><span class="label">GitHub</span>   
  Code</a>

![](images/packagereturns.jpg)

## Motivation

Return rates in E-commerce are significantly higher than in retail.
Especially when buying clothes online, customers expect free return
shipping. This comes with immense economic as well as ecological
costs.  
Therefore, online retailers are interested in predicting customer
returns at the time of their order. This enables them to better plan
ahead. It also creates the possibility to subtly nudge users to not
order a specific item they will return with a high probability.

## Data

The data originates from real orders in an online fashion store. It was
provided by Humboldt University in the context of this course project.
It contains information about the ordered items and their binary labels
describe if the customer returned a specific item or not. It also
contains user IDs that were generated for this dataset and that are not
traceable to the actual customers. The dataset is balanced such that
about half of the listed items were returned.

## Approach

Most of the work in this project encompasses data exploration, data
cleaning and feature engineering. Afterwards, the data will be
classified by experimenting with Logistic Regression and a Random Forest
Classifier, and then using the model that after parameter tuning
minimizes our loss function on test data.  
This problem has an asymmetric loss function that is based on the
following idea. If we incorrectly predict that a customer will return an
item, we nudge them to not buy it, resulting in a possible loss of
profit. If we fail to predict that a customer will return an item, the
store has to pay shipping costs as well as a fee for reconditioning the
item.  
To conveniently create features I wrote a custom scikit-learn
transformer. This makes the main part of the code in the notebook below
easier to read. It also enables us to effectively use it in pipelines.

{{< include-notebook file="content/posts/proj-returnpredictions/return_predictions_notebook.md" >}}

## Learnings

This project is a good example on how important thorough data
exploration and domain knowledge are in Data Science. It also showcases
how categories like color and size, whose naming and interpretation can
be subjective, are difficult to process in a model. Summing it up in one
sentence: Understand your data and what it presents, before you even
think about applying a model to it.

### Email

<davidsiriusschulte@gmail.com>

### Social

- <a href="https://github.com/davidschulte"
  class="icon brands alt fa-github"><span
  class="label">GitHub</span>/davidschulte</a>
- <a href="https://linkedin.com/in/davidsiriusschulte"
  class="icon brands fa-linkedin"><span
  class="label">GitHub</span>/davidsiriusschulte</a>

- Design: [HTML5 UP](https://html5up.net)
