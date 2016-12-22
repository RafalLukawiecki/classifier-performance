# Rafal Lukawiecki rafal@projectbotticelli.com https://projectbotticelli.com/datascience
# This code is licensed under Apache License Version 2.0, January 2004, http://www.apache.org/licenses/
# Please attribute to Rafal Lukawiecki when using or quoting

# Helper code for plotting and calculating various precision/recall metrics for an arbitrary classifier.
# Calculates the optimum prediction probability threshold given a ratio of the cost of a False Positive to a False Negative.



# Last updated on 10AUG16 # 02JUN16 fixed aspect ratios

# Example how to use the following function:
# Get the data into two vectors or columns of a data frame, one containing prediction probabilities (variable "scores") 
# and the other one with actual, known binary/boolean/dichotomous outcomes (variable "truth")

# For example, using the attached Weather Test Score data, taken from the excellent rattle package by Graham Wilson, you could do the following:
# preds1 <- read.csv("weather_test_score_idents.csv")
# table.at.threshold(preds1$rpart, preds1$RainTomorrow)

# What if I don't like getting rained on 4 times as much as I don't like carrying an unnecessary umbrella?
# table.at.threshold(preds1$rpart, preds1$RainTomorrow, cost.fp=1, cost.fn=4)

# Feel free to use the enclosed longer table, D1a, for further tests, ex:
# table.at.threshold(D1a$ScoredProbability, D1a$Label, pos.label = 1, neg.label = 0)
# or something more involved, like:
# table.at.threshold(D1a$ScoredProbability, D1a$Label, pos.label = 1, neg.label = 0, threshold=0.2, 
#                    cost.fp = 1, cost.fn = 100, label="VIP Buyer Classifier", table.resolution = 0.01)

# The above file has been derived from an educational machine learning data set "HappyCars" that you can get 
# when you participate in one of my classroom or online courses. If you want to learn practical data science with me, 
# have a look at https://projectbotticelli.com/courses 


table.at.threshold <- function(scores, truth, pos.label = TRUE, neg.label = FALSE, threshold=0.5, 
                               cost.fp = 1, cost.fn = 1, label="Confusion Matrix", table.resolution = 0.1) {
  # RLL: Helper function that prints a contingency table and precision-sensitivity metrics at a specified
  # cut-off threshold value of the predicted score
  
  require(ROCR, quietly = T)
  require(caret, quietly = T)
  
  if(pos.label %in% truth
     && neg.label %in% truth
     && mode(pos.label) == mode(neg.label)
     && mode(pos.label) == mode(truth)
  )
  {
    scores.at.threshold <- ifelse(scores >= threshold, pos.label, neg.label)
    
    warning <- min(scores.at.threshold) == max(scores.at.threshold)
    
    if(warning)
      print(paste("WARNING: At the threshold of", threshold, "there is no difference between predictions. Charts may be meaningless. Consider specifying a different (lower) threshold."))
    
    scores.f <- factor(scores.at.threshold, levels(factor(scores.at.threshold))[c(2, 1)])
    truth.f <- factor(truth, levels(factor(truth))[c(2, 1)])
    
    xtab <- table(scores.f, truth.f, dnn=c("Predictions","Outcomes"))
    
    print(xtab)
    
    pred <- prediction(scores, truth)
    perf.auc <- performance(pred, "auc")
    perf.roc <- performance(pred, "tpr", "fpr")
    perf.lift <- performance(pred, "tpr", "rpp")
    perf.pr <- performance(pred, "prec", "rec")
    perf.cost <- performance(pred, "cost", cost.fp = cost.fp, cost.fn = cost.fn)
    optimal.threshold <- pred@cutoffs[[1]][which.min(perf.cost@y.values[[1]])]
    
    print(paste(label, "at threshold", threshold))
    print(paste("At cost of FP", cost.fp, "FN", cost.fn, "optimal threshold would be", optimal.threshold))
    print(paste("AUC", perf.auc@y.values[[1]]))
    
    if(warning)
      print("Cannot print a complete confusion matrix or its threshold-specific metrics")
    else
      print(confusionMatrix(xtab))
    
    par(pty="s") # Make square plots
    plot(perf.roc)
    title(main="ROC",
          sub=paste("RL", format(Sys.time(), "%Y-%b-%d %H:%M:%S")))
    plot(perf.cost)
    title(main="Cost Chart",
          sub=paste("RL", format(Sys.time(), "%Y-%b-%d %H:%M:%S")))
    plot(perf.pr)
    title(main="Precision-Recall",
          sub=paste("RL", format(Sys.time(), "%Y-%b-%d %H:%M:%S")))
    plot(perf.lift, xlab="Caseload (%)", ylab="Target Population (%)")
    title(main="Lift Chart (Cumulative Gain)",
          sub=paste("RL", format(Sys.time(), "%Y-%b-%d %H:%M:%S")))
    
    
    print(performance.by.probability(scores, truth == pos.label, table.resolution))
  } else
  {
    print("Please specify the label values for the positive and negative classes, making sure they match what is in the 'truth' parameter. Currently I see them as:")
    print(mode(truth))
    print(mode(pos.label))
  }
}


# Produce a table summarising precision/recall and other performance metrics at different cut-off points of predicted probability
# Increase resolution to 0.01 or even more if you wish to use the resulting data for curve plotting, however, it would be much 
# easier to plot curves, such as ROC, Lift, P-R etc using the preceding, above function, which uses a neat, ready-made package called ROCR.

# If you get errors, make sure the truth is a numeric vector of 0 and 1s, or a factor or anything else that sensibly coerces to it.

performance.by.probability <- function(scores, truth, resolution = 0.1) {
  
  require(dplyr, quietly = T)
  
  # Make a data frame from the supplied probabilities and known outcomes (truths)    
  p <- data_frame(probability = scores, outcome = truth)
  
  sample.size <- nrow(p) # Number of rows
  
  # Assign each row to a probability bin corresponding to the desired resolution, which will dictate the number of resulting rows.
  # E.g, resolution = 0.1 produces 10 rows.
  p$bin <- cut(p$probability, breaks = seq(0, 1, resolution))
  
  # Create a contingency table, that is tabulate how many positive and negative outcomes there are in each probability bin
  t <- as.data.frame.matrix(table(p$bin, p$outcome))
  
  # Rename the contingency table headings
  names(t) <- c("Negative.Examples", "Positive.Examples")   # This may work the wrong way round in case you provide anything other than 0-1 as truth
  
  # Include the name of the probability bin as a column of the resulting data frame
  t$Probability.Bin <- rownames(t)
  
  # This uses dplyr hence the chaining %>% pipe syntax, which simply passes the result of one data frame operation to the next one
  # We start with t, the contingency table of outcome counts binned by probability, and...
  performance.table <- t %>% 
    arrange(desc(Probability.Bin)) %>%    # Sort by probability, *descending*
    select(Probability.Bin, Positive.Examples, Negative.Examples, everything()) %>%    # Reorder columns, to a more commonly used layout
    mutate(    # This simply adds new calculated columns, letting us refer to columnar and tabular functions and other, just-calculated columns
      
      # Calculate the elementary, bin-specific confusion matrices, ie. numbers of True Positives etc for each, *cumulative* probability bin
      TP = cumsum(Positive.Examples),   # Cumulative sum, from largest to smallest probability, thanks to earlier sorting
      FP = cumsum(Negative.Examples),
      TN = sum(Negative.Examples) - FP,
      FN = sum(Positive.Examples) - TP,
      
      # The most important 3 performance metrics
      # Precision = TP / (TP + FP),  # Aka Positive predictive value (PPV)
      # Naive version above produces division by zero NaNs, the following calculates Precision of 1 at the point
      # where all cases have already been "seen", that is towards the end of the ROC curve.
      Precision = ifelse(TP == 0 & FP == 0, 1, TP / (TP + FP)),  # Aka Positive predictive value (PPV)
      Recall = TP / (TP + FN),     # Aka *Sensitivity* or True positive rate (TPR)
      F1.Score = 2 * Precision * Recall / (Precision + Recall),
      
      # The less important, but also useful metrics 
      Accuracy = (TN + TP) / sample.size,   # Careful: can be misleading
      FPR = FP / (TN + FP),    # False positive rate, aka fall-out, useful for ROC charts, simply plot x=FPR y=Recall
      TNR = TN / (TN + FP),    # True negative rate, aka *Specificity*
      NPV = ifelse(TN == 0 & FN == 0, 1, TN / (TN + FN)),    # Negative predictive value
      
      # The two metrics useful for building Lift aka Cumulative Gain Charts. Simply plot x=RPP y=Cumulative.Gain
      RPP = (TP + FP) / sample.size,    # Rate of positive predictions
      Cumulative.Gain = TP / sum(Positive.Examples)    # Aka gain score
    )
  
  # Examples of ROC and Lift plots, but bear in mind, it is better to use the ROCR package, as shown in the function table.at.threshold, above.
  # require(ggplot2, quietly = T)
  # print(ggplot(data=performance.table, aes(FPR, Recall)) + geom_line() + coord_fixed(xlim = c(0,1), ylim = c(0,1)) + ggtitle("ROC Curve"))
  # print(ggplot(data=performance.table, aes(RPP, Cumulative.Gain)) + geom_line() + coord_fixed(xlim = c(0,1), ylim = c(0,1)) + ggtitle("Lift (Cumulative Gain) Chart"))
  
  return(performance.table)
}
