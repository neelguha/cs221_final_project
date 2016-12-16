# Predicting Authorship of Reddit Comments 

## Annie Hu, Cindy Wang, Neel Guha 


This directory contains our scripts and data for this project. Below is a description of how to run the experiments. Because we took samples, replicating the entire process may lead to results that vary slightly from those described in the paper.



### Data Pre-processing

Commands for data processing/feature generation.

1. python create_sample.py 

Samples a set of users and a set of comments from those users from the file data/politics.tsv. Writes the results out to sampled_users.tsv

2. python feature_generator.py 

For each comment in our sample, generate the corresponding feature vector and write out the feature vector to data/sampled_users.tsv.


### Binary Classification 

1. python single_user_classification_precomputed_features.py

Runs binary classification on all users and reports the average precision/recall. Adjusting flags in the script control which model (Naive Bayes, SVM, etc) is used.  

2. python binary_baseline.py 

Runs and reports the results of the binary baseline model. 

3. python binary_oracle.py 

Runs and reports the results of the binary oracle. 

### Multiclass Classification 

1. python multi_user_classification.py 

Runs multiclass classification on all users and reports the micro/macro precision/recall. Adjusting flags in the script control which model (Naive Bayes, etc) is used.  

2. python multi_baseline.py 

Runs and reports the results of the multiclass baseline model. 

3. python multi_oracle.py 

Runs and reports the results of the multiclass oracle. 

### Unsupervised Methods

1. python unsupervised_clustering.py 

Runs k-means on all users and reports the micro/macro precision/recall. Adjusting flags in the script control which model (Naive Bayes, etc) is used.  

2. python unsupervised_clustering_baseline.py 

Runs and reports the results of the unsupervised baseline model. 

3. python unsupervised_clustering_oracle.py 

Runs and reports the results of the unsupervised oracle. 


## Data Files 

1. data/politics.tsv 
Contains all comments on /r/politics from 2014. Data is in the form of

"<subreddit_name>  <time_stamp>  <subreddit_id>  <comment_id>  <parent_comment_id> <author_name> <score> <???> <thread/link_id> <text>"


