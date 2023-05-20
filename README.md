# MultiLingual_Toxic_Comment_Classification

---

## Problem Statement and Description

It only takes one toxic comment to sour an online discussion. Identifing the toxicity in online conversations, where toxicity is defined as anything `rude, disrespectful or otherwise likely to make someone leave a discussion`, in an automated way using machine learning, at very early stage, is one way to protect voices in online conversations.

## The Multi-lingual Toxic Comment Classification project is a powerful tool for content platforms operating in various domains, including social media, news websites, forums, and on the internet. By leveraging the latest Natural Language Processing techniques, it automatically detects and filters toxic comments in multiple languages, addressing the pressing issues of cyberbullying, hate speech and harassment.

---

## Dataset Description

The dataset has been taken from `Kaggle` competition organized by the `Jigsaw/Conversation AI` team. The dataset contains several files:

- jigsaw-toxic-comment-train.csv - data from our [first competition](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge). The dataset is made up of English comments from Wikipedia’s talk page edits.
- jigsaw-unintended-bias-train.csv - data from our [second competition](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification). This is an expanded version of the Civil Comments dataset with a range of additional labels.
- sample_submission.csv - a sample submission file in the correct format.
- test.csv - comments from Wikipedia talk pages in different non-English languages (Spanish, Portuguese, Italian, Turkish, French, Russian).
- validation.csv - comments from Wikipedia talk pages in different non-English languages.

Columns

- id - identifier within each file.
- comment_text - the text of the comment to be classified.
- lang - the language of the comment.
- toxic - whether or not the comment is classified as toxic. (Does not exist in test.csv.)

Here, we're predicting the probability that a comment is `toxic`. A toxic comment would receive a `1.0`. A benign, non-toxic comment would receive a`0.0`. In the test set, all comments are classified as either a `1.0` or `0.0`.
