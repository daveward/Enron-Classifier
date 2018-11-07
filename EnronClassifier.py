#########
# 

# Date: 09 December 2017
# Author: David Ward
#
#########   

% matplotlib inline
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn import model_selection
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV
import seaborn as sns
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
import itertools
import shutil
from random import randint

from collections import Counter
from sklearn.feature_extraction import text
import re
import os
import pickle


""" NOTE:

 sourceDIR and destFolder must be changed to your appropriate local directory
 where the PRE directory is located
"""

sourceDir = "C:\\Users\\user\\Documents\\*****\\Enron\\enron\\pre\\" 
destFolder = "C:\\Users\\user\\Documents\\******\\Enron\\enron\\pre\\"

hamDir = destFolder + "HAM\\"
spamDir = destFolder + "SPAM\\"

subjectList = [] # Ham subjects
bodyList = [] #Ham Body
spamSubject = [] #Spam subject list
spamBody = [] # Spam Body List
validationSPAMBody = [] # Validation Set SPAM Body list
validationHAMBody = [] # Validation Set HAM Body list
validationSPAMSubject = [] # Validation Set SPAM Subject List
validationHAMSubject = [] # Validation Set HAM Subject List


# List variables are created to store our subject, body (both SPAM and HAM) for our Training, Test and Validation sets.

                                                        
def getHamFiles(sourceDir, destDir):
    for i in range (1,7):
        src_files = os.listdir(sourceDir +"enron" +  str(i) + "\\ham\\")
        for file in src_files:
            full_file_name = os.path.join((sourceDir + "enron" + str(i) + "\\ham\\"), file)
            if (os.path.isfile(full_file_name)):
                if not os.path.exists(destFolder + "HAM\\"):
                    os.makedirs(destFolder+"HAM\\")
                if (os.path.isfile(destFolder+"HAM\\"+file)):
                    pass
                else:
                    shutil.move(full_file_name, (destFolder+"HAM\\"))


def getSpamFiles(sourceDir, destDir):
    for i in range (1,7):
        src_files = os.listdir(sourceDir +"enron"  + str(i) + "\\spam\\")
        for file in src_files:
            full_file_name = os.path.join((sourceDir + "enron" +str(i) + "\\spam\\"), file)
            if (os.path.isfile(full_file_name)):
                if not os.path.exists(destFolder + "SPAM\\"):
                    os.makedirs(destFolder+"SPAM\\")
                if (os.path.isfile(destFolder+"SPAM\\"+file)):
                    pass
                else:
                    shutil.move(full_file_name, (destFolder+"SPAM\\"))


""" 
We call both functions and supply the appropriate directories.

In order to perform a validation test on our final classifier, we must select a sufficient number of SPAM and HAM mails for our validation data set.

getRangeFiles generates a 250 item array of random integers between 0 and 16500. An array (in this instance 250) of HAM and SPAM emails will be selected based upon their index number. The index number will be between 0 and 16500.

"""
                                                                                            
def getRangeFiles(fRange):
    fileIndex = []
    for i in range (0, fRange):
        if (randint(0,16500)) not in fileIndex:
            fileIndex.append(randint(0,16500))
    return ((fileIndex))


# makeValidation iterates through the specified HAM or SPAM directory and moves files with indexes specified in fileIndex to their respective subdirectories in the validation directory.

def makeValidation(fType):
    src_files = os.listdir(destFolder + fType + "\\")
    if not os.path.exists(destFolder + fType + "\\"):
        os.makedirs(destFolder + fType + "\\")
    for i in getRangeFiles(250):
        full_file_name = os.path.join((destFolder + fType + "\\"), src_files[i])
        if (os.path.isfile(full_file_name)):
            if not os.path.exists(destFolder + "validation" + "\\" + fType + "\\"):
                os.makedirs(destFolder + "validation" +"\\" +  fType + "\\")
            shutil.move(full_file_name, (destFolder+"validation" +"\\" + fType +"\\"))
            
            
def getValidationMails(fType):

    validDir = (destFolder + "\\validation" + "\\" + fType + "\\")
    for subdir, dirs, files in os.walk(validDir):
        for file in files:
            string = open(validDir+file).read()
            new_str = re.sub('[^a-zA-Z0-9\n\:]', ' ', string)
            open(validDir+file, 'w').write(new_str)
            
    for files in os.walk(validDir):
        for file in files[2]:
            with open (validDir+file, 'r') as f:
                first_line = f.readline().rstrip()
                if fType == "HAM":
                    validationHAMSubject.append(first_line)
                if fType == "SPAM":
                    validationSPAMSubject.append(first_line)
                body_line = f.read().split('\n')
                if body_line == '':
                    pass
                if body_line == ' ':
                    pass
                if body_line == '[]':
                    pass
                if fType == "HAM":
                        validationHAMBody.append(body_line)                        
                if fType == "SPAM":
                        validationSPAMBody.append(body_line)
                        
# We then call the functions and provide the appropriate classes.

makeValidation("HAM")
makeValidation("SPAM")
getValidationMails("HAM")
getValidationMails("SPAM")

# Similarly with getValidationMails, we do the same for HAM and SPAM to clean the mails and populate the appropriate arrays.


# Prepare ham files

for subdir, dirs, files in os.walk(hamDir):
    
    for file in files:
        
        string = open(hamDir+file).read()
        new_str = re.sub('[^a-zA-Z0-9\n\:]', ' ', string)
        open(hamDir+file, 'w').write(new_str)

        
for files in os.walk(hamDir):
    for file in files[2]:
        with open (hamDir+file, 'r') as f:
            first_line = f.readline().rstrip()
            subjectList.append(first_line)
            body_line = f.read().split('\n')
            if body_line == '':
                pass
            if body_line == ' ':
                pass
            if body_line == '[]':
                pass
            else:
                bodyList.append(body_line)


# Prepare SPAM files

for subdir, dirs, files in os.walk(spamDir):
    for file in files:
        string = open(spamDir+file, encoding='latin-1').read()
        
        new_str = re.sub('[^a-zA-Z0-9\n\:]', ' ', string)
        
        open(spamDir+file, 'w').write(new_str)

for files in os.walk(spamDir):
        for file in files[2]:
            with open (spamDir+file, 'r', encoding='latin-1') as f:
                first_line = f.readline().rstrip()
                spamSubject.append(first_line)
                body_line = f.read().split('\n')
                if body_line == '':
                    pass
                if body_line == ' ':
                    pass
                if body_line == '[]':
                    pass
                else:
                    spamBody.append(body_line)
                    
                    
# dataframes must be created from the data stored

HAM = 'HAM'
SPAM = 'SPAM'

enronHAM = pd.DataFrame({'Subject': subjectList, 'Body': bodyList, 'Classification': HAM})
enronSPAM = pd.DataFrame({'Subject': spamSubject, 'Body': spamBody, 'Classification': SPAM})
enronValidationSPAM = pd.DataFrame({'Subject': validationSPAMSubject, 'Body': validationSPAMBody, 'Classification': SPAM})
enronValidationHAM = pd.DataFrame({'Subject': validationHAMSubject, 'Body': validationHAMBody, 'Classification': HAM})

# Clean the dataframes

enronSPAM['Body'] = enronSPAM['Body'].dropna(how='any')
enronHAM['Body'] = enronHAM['Body'].dropna(how='any')
enronValidationSPAM['Body'] = enronValidationSPAM['Body'].dropna(how='any')
enronValidationHAM['Body'] = enronValidationHAM['Body'].dropna(how='any')


# Initial statistics are performed

# HAM statistics

print ("Mail sizes")
bodyLengthHAM = []
for mails in (enronHAM.Body.str.len()):
    bodyLengthHAM.append(mails)

bodyLengthSPAM = []
for mails in (enronSPAM.Body.str.len()):
    bodyLengthSPAM.append(mails)

hamMailSizes = pd.DataFrame({'HAM':bodyLengthHAM})


hamMailSizes.HAM = hamMailSizes.HAM.astype(int)
print ("Total HAM mail size:", hamMailSizes.HAM.count())
print ("Mean HAM mail size:", hamMailSizes.HAM.mean())

sns.boxplot(hamMailSizes.HAM).set_title("HAM mails")

"""
Mail sizes
Total HAM mail size: 16301
Mean HAM mail size: 25.59168149193301
"""

# Spam Mail statistics
bodyLengthSPAM = []
for mails in (enronSPAM.Body.str.len()):
    bodyLengthSPAM.append(mails)
spamMailSizes = pd.DataFrame({'SPAM':bodyLengthSPAM})
print ("Mean spam mail size:", spamMailSizes.SPAM.mean())
print ("Total Spam Mails: ", spamMailSizes.SPAM.count())

spamMailSizes.SPAM = spamMailSizes.SPAM.astype(int)
sns.set(style='ticks')
sns.boxplot(spamMailSizes.SPAM).set_title("All SPAM mails")


"""
Both the SPAM and HAM datasets now need to be combined to become the full Enron dataset.
This will allow us to split the data into training and test sets later on. 
The validation sets are also combined to build the separate validation set for use on our final chosen model.
"""

combinedEnron = [enronSPAM, enronHAM]

fullEnron = pd.concat(combinedEnron)

validationCombined = [enronValidationHAM, enronValidationSPAM]

validationSet = pd.concat(validationCombined)


"""
To prevent bias in our training and test sets and to ensure the data is distributed sufficiently, 
the dataframes must be reindexed so the data can be split.
"""

#rebuild the index
fullEnron = fullEnron.reset_index(drop=True) 
fullEnron['Body'] = fullEnron.Body.apply(','.join)
fullEnron = fullEnron.ix[fullEnron['Body'] != ""]
fullEnron = fullEnron.reindex(np.random.permutation(fullEnron.index))

fullEnron['Body'] = fullEnron['Body'].str.replace(',', ' ')

# Do the same for the validation set
validationSet = validationSet.reset_index(drop=True)
validationSet['Body'] = validationSet.Body.apply(','.join)
validationSet = validationSet.ix[validationSet['Body'] != ""]
validationSet = validationSet.reindex(np.random.permutation(validationSet.index))


# Create training set
trainingEnron, testEnron = train_test_split(fullEnron, test_size=0.3)

# Training and Test set stats

fig, ax = plt.subplots(1,2)
sns.countplot(trainingEnron.Classification, ax=ax[0])
sns.countplot(testEnron.Classification, ax=ax[1])

ax[0].set_title("Training")
ax[1].set_title("Test")
fig.tight_layout()


# Training dataset
print ("Total training mails: ", trainingEnron.Body.count())
trainBodyLength = []
for mails in (trainingEnron.Body.str.len()):
    trainBodyLength.append(mails)
trainingMailSizes = pd.DataFrame({'Body':trainBodyLength})

print ("Mean training mail size: ", trainingMailSizes.Body.mean())


# Test Dataset
print ("Total testing mails: ", testEnron.Body.count())
testBodyLength = []
for mails in (testEnron.Body.str.len()):
    testBodyLength.append(mails)
testMailSizes = pd.DataFrame({'Body':testBodyLength})

print ("Mean test mail size: ", testMailSizes.Body.mean())


# Count our words in our training set
results = Counter()
countV = CountVectorizer(stop_words='english')
countV.fit_transform(trainingEnron.Body)
trainingEnron.Body.str.lower().str.split().apply(results.update)

# Return the top 20 words
topTwenty = (results.most_common(20))
print (topTwenty)


# Examine the top 50, which most seem to be stop words. Append top 50 to stop_words.
dictAr = (results.most_common(50))
new_words = []
for key, value in dictAr:
    new_words.append(key)

stop_words = text.ENGLISH_STOP_WORDS.union(new_words)



# Multinomial Naive Bayes Pipeline with stop word removal

MNBpipeline = Pipeline([
    ('vectorizer',  CountVectorizer(stop_words=stop_words)),
    ('classifier',  MultinomialNB()) ])
    
MNBpipeline.fit(trainingEnron.Body.values, trainingEnron.Classification.values)

MNBPipelineClass =  (MNBpipeline.predict(testEnron.Body.values))
print ("Multinomial BP: ", metrics.accuracy_score(testEnron.Classification.values, MNBPipelineClass))

"""
The following classifiers have been tested.

Multinomial Naive Bayes - with CountVectorizer and a MultiNomial Naive Bayes Classifier.
MultiNomial Naive Bayes - with TfidfVectorizer and a Multinomial Naive Bayes Classifer.
Support Vector Classifier - with TfidfVectorizer and a LinearSVC classifier.
Support Vector Classifier - with CountVectorizer and a LinearSVC classifier.
Support Vector Classifier - with HashingVectorizer and a LinearSVC classifier.
Linear Regression Classifier - with TfidfVectorizer and a LinearRegression classifier.
Variations of these classifiers have been implemented, with stop word removal and document frequency variations.
"""


# Multinomial Naive Bayes Pipeline with stop word removal

MNBpipeline = Pipeline([
    ('vectorizer',  CountVectorizer(stop_words=stop_words)),
    ('classifier',  MultinomialNB()) ])
    
MNBpipeline.fit(trainingEnron.Body.values, trainingEnron.Classification.values)

MNBPipelineClass =  (MNBpipeline.predict(testEnron.Body.values))
print ("Multinomial BP: ", metrics.accuracy_score(testEnron.Classification.values, MNBPipelineClass))


# MNB with TFIDF with sublinear and max document frequency

MNBTFpipeline = Pipeline([
    ('vectorizer',  TfidfVectorizer( sublinear_tf=True, max_df=0.69
                                 )),
    ('classifier',  MultinomialNB()) ])
    
MNBTFpipeline.fit(trainingEnron.Body.values, trainingEnron.Classification.values)
MNBTFPipelineClass =  (MNBTFpipeline.predict(testEnron.Body.values))
MNFIT = MNBTFpipeline.fit(trainingEnron.Body.values, trainingEnron.Classification.values)

print ("Multinomial BP with TF: ", metrics.accuracy_score(testEnron.Classification.values, MNBTFPipelineClass))



# SVM with TFIDF and Calibrated LinearSVC

linSVCC = CalibratedClassifierCV(LinearSVC())
LinearSVCpipeline = Pipeline([
    ('vectorizer',  TfidfVectorizer(sublinear_tf=True, max_df=0.69, stop_words=stop_words)),
    ('classifier',  linSVCC) ])

LinearSVCpipeline.fit(trainingEnron.Body.values, trainingEnron.Classification.values)
LinearSVCpredictClass =  (LinearSVCpipeline.predict(testEnron.Body.values))
LinearSVCprob = LinearSVCpipeline.fit(trainingEnron.Body.values, trainingEnron.Classification.values)

LinearSVCFit = (LinearSVCpipeline.fit(trainingEnron.Body.values, trainingEnron.Classification.values))
print ("LinearSVC Score: ", metrics.accuracy_score(testEnron.Classification.values, LinearSVCpredictClass))


# SVM Classifier with Linear Classification and hinge loss

SVCCVpipeline = Pipeline([
    ('vectorizer',  CountVectorizer()),
    ('classifier',  LinearSVC(loss='hinge')) ])

SVCCVpipeline.fit(trainingEnron.Body.values, trainingEnron.Classification.values)
SVCCVpredictClass =  (SVCCVpipeline.predict(testEnron.Body.values))
print ("SVCCV Score: ", metrics.accuracy_score(testEnron.Classification.values, SVCCVpredictClass))


# SVM with HashingVectorizer and LinearSVC
SVHVpipeline = Pipeline([
        ('vectorizer', HashingVectorizer()),
        ('classifier', LinearSVC()) ])
    
    
    
SVHVpipeline.fit(trainingEnron.Body.values, trainingEnron.Classification.values)

SVHVpredictClass = SVHVpipeline.predict(testEnron.Body.values)
print ("SVHV Score: ", metrics.accuracy_score(testEnron.Classification.values, SVHVpredictClass))
SVHV Score:  0.987519025875

# SVM Pipeline with CountVectorizer, stop word removal, LinearSVC and hinge loss.

SVMpipeline = Pipeline([
        ('vectorizer', CountVectorizer(stop_words=stop_words)),
        ('classifier', LinearSVC(loss='hinge')) ])
    
    
    
SVMpipeline.fit(trainingEnron.Body.values, trainingEnron.Classification.values)

SVMpredictClass = SVMpipeline.predict(testEnron.Body.values)
print ("SVM Score: ", metrics.accuracy_score(testEnron.Classification.values, SVMpredictClass))


# Logistic Regression with TFIDFvectorizer

LRpipeline = Pipeline([
        ('vectorizer', TfidfVectorizer()),
        ('classifier', LogisticRegression())
    ])


LRpipeline.fit(trainingEnron.Body.values, trainingEnron.Classification.values)

LRpredClass = LRpipeline.predict(testEnron.Body.values)
print ("Linear Regression: ", metrics.accuracy_score(testEnron.Classification.values, LRpredClass))

"""
By the classifier results, the SVM with TFIDF and a Calibrated LinearSVC with stop word removal and document frequency is the clear winner. 
The classifiers are evaluated again with a confusion matrix.
"""


## Code modified from example given on SKLEARN documentation

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    
class_names = ['HAM', 'SPAM']

# The classifiers are now evaluated on a confusion matrix


# Linear Regression plot
lrMetrics = (metrics.confusion_matrix(testEnron.Classification.values, LRpredClass))
plt.figure()
plot_confusion_matrix(lrMetrics, classes=class_names, normalize=True,
                      title='Normalized confusion matrix: Linear Regression')


# SVM Pipeline with CountVectorizer, stop word removal, LinearSVC and hinge loss.

svmMetrics = (metrics.confusion_matrix(testEnron.Classification.values, SVMpredictClass))
plt.figure()
plot_confusion_matrix(svmMetrics, classes=class_names, normalize=True, title='Normalized confusion matrix: SVM')


# Multinomial Naive Bayes Pipeline with stop word removal

MNBmetrics = (metrics.confusion_matrix(testEnron.Classification.values, MNBPipelineClass))
plt.figure()

plot_confusion_matrix(MNBmetrics, classes=class_names, normalize=True, title='Normalized confusion matrix: MNBmetrics')


# MNB with TFIDF with sublinear and max document frequency

MNBTFmetrics = (metrics.confusion_matrix(testEnron.Classification.values, MNBTFPipelineClass))
plt.figure()

plot_confusion_matrix(MNBTFmetrics, classes=class_names, normalize=True, title='Normalized confusion matrix: MNBTFmetrics')


# Linear SVC TF Plot
LinearSVCMetrics = (metrics.confusion_matrix(testEnron.Classification.values, LinearSVCpredictClass))
plt.figure()
plot_confusion_matrix(LinearSVCMetrics, classes=class_names, normalize=True, title='Normalized confusion matrix: LinearSVC')



# The counts of SPAM and HAM emails in our training set

print ("SPAM / HAM", trainingEnron.Classification.value_counts())


# Classification Report for our best classifier

print (metrics.classification_report(testEnron.Classification.values, LinearSVCpredictClass ))
print ("mean classification accuracy")
print (np.mean(testEnron.Classification.values == LinearSVCpredictClass))


# The precision recall curve for our most accurate classifer

lb = LabelBinarizer()


trueValues = lb.fit_transform(testEnron.Classification.values)
LabelBinarizer(neg_label=0, pos_label=1, sparse_output=False)
predValues = lb.fit_transform(LinearSVCpredictClass)
precision, recall, _ = metrics.precision_recall_curve(trueValues, predValues)
average_precision = average_precision_score(trueValues, predValues)

print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))
plt.figure()
plt.step(recall, precision, color='b', alpha=0.3,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.3,
                 color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.2])
plt.xlim([0.0, 1.2])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
          average_precision))

print ("roc auc score: ", roc_auc_score(trueValues, predValues))

# The classifiers have been assessed and the most accurate has been chosen and it's metrics detailed. 
# The classifier can now be saved to disk. The saved classifier model will then be loaded and validated with our validation set from earlier.


# modify this variablefor a local directory
savedModel = "C:\\Users\\user\\Documents\\...\\Enron\\ourModel.sav" 

pickle.dump(LinearSVCpipeline, open(savedModel, 'wb')) # save the model to disk

assignmentModel = pickle.load(open(savedModel, 'rb')) # reload the saved model

print ("loaded pickle: ", assignmentModel) # details of the model

# Using the validation dataframe from earlier, the saved model can then be used to predict on the unseen data.        

validationResult = assignmentModel.predict(validationSet.Body.values) 

print ("--------------")
print ("Model score ", metrics.accuracy_score(validationSet.Classification.values, validationResult))
print ("--------------")
print (metrics.classification_report(validationSet.Classification.values, validationResult ))
print ("Mean classification accuracy")
print (np.mean(validationSet.Classification.values == validationResult))


class_names = ["HAM", "SPAM"]
FinalModelMetrics = (metrics.confusion_matrix(validationSet.Classification.values, validationResult))
plt.figure()
plot_confusion_matrix(FinalModelMetrics, classes=class_names, normalize=True, title='Normalized confusion matrix: LinearSVC - Final Model')


lb = LabelBinarizer()


trueValues = lb.fit_transform(validationSet.Classification.values)
LabelBinarizer(neg_label=0, pos_label=1, sparse_output=False)
predValues = lb.fit_transform(validationResult)
precision, recall, _ = metrics.precision_recall_curve(trueValues, predValues)
average_precision = average_precision_score(trueValues, predValues)

print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))
plt.figure()
plt.step(recall, precision, color='b', alpha=0.3,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.3,
                 color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.2])
plt.xlim([0.0, 1.2])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
          average_precision))

print ("ROC AUC score: ", roc_auc_score(trueValues, predValues))


# Validation Set Stats
sns.countplot(validationSet.Classification).set_title("Validation Set")


validationBodyLength = []
for mails in (validationSet.Body.str.len()):
    validationBodyLength.append(mails)
validationMailSizes = pd.DataFrame({'Body':validationBodyLength})

print ("Mean validation mail body length: ", validationMailSizes.Body.mean())
print ("Total validation set size: ", validationMailSizes.Body.count())