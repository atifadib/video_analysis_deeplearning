# Activity Recognition using Neural Networks
Developing a city wide surveillance framework for live stream video 

Dataset: http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/
We have cascaded structure of classifiers, we generate a new dataset from the dataset mentioned above by extract region of interest
specific features.
For each of the 51 classes of actions, we will then build three different set of classifiers each fine tuned to recognize a specific set of
activities within a subclass among the 51 actions.

# Unsupervised Activity differentiator 
Building a temporal sense of the motion about the object is imperative for any activity classification model to be robust and work in all conditions.
We build a temporal sense of activity from the dataset using RCNNs, we feed the entire HMDB dataset through our YOLO model to identify person/people in that object and build a sequence of frames pertaining to that class.
We will separately train our model on this new dataset.
*Retraining the activity differentiator

