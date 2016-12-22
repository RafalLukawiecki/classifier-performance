# Plot a classifier's performance in R

Plot ROC, precision-recall, cost and lift curves, calculate optimum probability threshold, print a confusion matrix, 
and additional metrics, for a machine learning classifier.

## Motivation

Whilst testing classifier performance, it is helpful to compare a number of classification accuracy visualisations. This code
displays several of them, and it helps to tabulate accuracy data to prepare your own plots, if you wish to do so. The only required input
is a vector of known outcomes and a vector of predicted probabilities. As a bonus, this code will look up the optimum
prediction probability threshold given a ratio of the cost of a False Positive to a False Negative.

_Last updated on 10AUG16, fixed aspect ratios_

## How to use

Either open the entire R project in RStudio, or simply use script file `classifier-performance.R`.
Get the data into two vectors or columns of a data frame, one containing prediction probabilities (variable "scores") and the other one with actual, known binary/boolean/dichotomous outcomes (variable "truth").

For example, using the attached Weather Test Score data, taken from the excellent _rattle_ package by Graham Wilson, you could do the following:

```{r}
preds1 <- read.csv("weather_test_score_idents.csv")
table.at.threshold(preds1$rpart, preds1$RainTomorrow)
```

What if I don't like getting rained on 4 times as much as I don't like carrying an unnecessary umbrella?

```{r}
table.at.threshold(preds1$rpart, preds1$RainTomorrow, cost.fp=1, cost.fn=4)
```

Feel free to use the enclosed, longer table `D1a` for further tests, eg.:

```{r}
table.at.threshold(D1a$ScoredProbability, D1a$Label, pos.label = 1, neg.label = 0)
```

or something more involved, like:

```{r}
table.at.threshold(D1a$ScoredProbability, D1a$Label, pos.label = 1, neg.label = 0, threshold=0.2,
                   cost.fp = 1, cost.fn = 100, label="VIP Buyer Classifier", table.resolution = 0.01)
```

`D1a` has been derived from an educational machine learning data set _HappyCars_ that you can get
when you participate in one of my classroom or online courses. If you want to learn practical data science with me,
have a look at <https://projectbotticelli.com/courses>

_Rafal_
