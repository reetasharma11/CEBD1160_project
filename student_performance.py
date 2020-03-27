"""
Project:  Student performance analysis

"""

# Import necessary libraries and load data sets to pandas DataFrame.
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import LinearSVR
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn import neighbors
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from warnings import filterwarnings

filterwarnings('ignore')

# Load data for subject "Math".
math = pd.read_csv('student/student-mat.csv', sep=";")
print(math.shape)  # (395, 33)
# add an extra column ['subject] to math DataFrame.
math.insert(1, 'subject', ['math'] * 395)

# Load data for subject "Portuguese".
portuguese = pd.read_csv('student/student-por.csv', sep=";")
print(portuguese.shape)  # (649, 33)
# add an extra column ['subject] to portuguese dataframe.
portuguese.insert(1, 'subject', ['por'] * 649)

# Concatenate both DataFrame vertically
students = pd.concat([math, portuguese])

# Check and make sure the concatenation is correct
assert math.shape[0] + portuguese.shape[0] == students.shape[0], 'merge error'
assert math.shape[1] == portuguese.shape[1] == students.shape[1], 'merge error'

# Check the DataFrame
print(students.shape)
print(students.head())  # column 'subject' has been inserted.
print(students.describe())
print(students.info())  # No data missing, but some columns data type are object, data cleansing needed before ML.

# Sort out the all the column names with data type object
text_columns = []
dataTypeDict = dict(students.dtypes)
for col in dataTypeDict:
    if dataTypeDict[col] == 'O':
        text_columns.append(col)
print(text_columns)

# convert all the two-answers categorical features to integers: (Mjob, Fjob, reason, guardian, needs one-hot-encoding
# method to convert into numerical data)
students['school'] = students['school'].map({'GP': 0, "MS": 1})
students['subject'] = students['subject'].map({'math': 0, "por": 1})
students['sex'] = students['sex'].map({'F': 0, "M": 1})
students['address'] = students['address'].map({'U': 0, "R": 1})
students['famsize'] = students['famsize'].map({'GT3': 0, "LE3": 1})
students['Pstatus'] = students['Pstatus'].map({'A': 0, "T": 1})
students['schoolsup'] = students['schoolsup'].map({'no': 0, "yes": 1})
students['famsup'] = students['famsup'].map({'no': 0, "yes": 1})
students['paid'] = students['paid'].map({'no': 0, "yes": 1})
students['activities'] = students['activities'].map({'no': 0, "yes": 1})
students['nursery'] = students['nursery'].map({'no': 0, "yes": 1})
students['higher'] = students['higher'].map({'no': 0, "yes": 1})
students['internet'] = students['internet'].map({'no': 0, "yes": 1})
students['romantic'] = students['romantic'].map({'no': 0, "yes": 1})

# Recheck the dtypes
print(students.info())

# Data visualization
os.makedirs('plots/visual', exist_ok=True)
os.makedirs('plots/ML', exist_ok=True)

# Plotting the heatmap (missing Mjob, Fjob, reason, guardian)
_, ax = plt.subplots(figsize=(25, 25))
cmap = sns.diverging_palette(220, 10, as_cmap=True)  # color map

# Numpy’s tril() function to extract Lower Triangle Matrix
df_lt = students.corr().where(np.tril(np.ones(students.corr().shape)).astype(np.bool))

# plotting the heatmap
sns.heatmap(data=df_lt,
            cmap=cmap,
            square=True,
            cbar_kws={'shrink': .6},
            annot=True,
            annot_kws={'fontsize': 10},
            ax=ax
            )
plt.savefig(f'plots/visual/heatmap.png')
plt.close()

"""
Interesting findings of Heatmap: 
1. G1 and G2 and failures are the most 3 related features to final grade G3. 
2. Mother's education is very much related with father's education. 
3. Beside the G1 and G2 and failures, we can see the other Top10 influence factors to target G3 are: higher:0.24, 
Medu:0.2, Fedu:0.16, studytime:0.16, age:-0.13, Dalc:-0.13, address:-0.12, Walc: -0.12, internet:0.11, traveltime: -0.1.
4. walc and goout has high relation. 
5. traveltime and address has high relation. 
6. paid and subject has high negtive relation. 
7. Internet has a positive relation to target G3. 
8. sex, Pstatus, schoolsup, famsup, paid, nursery, romantic, famrel, health, absences (These factors are surpriseingly 
showing us the grade is not much related to them, which is contrary to our usual perception: family support, school 
support or extra classes paid should greatly help to improve grades but not as hoped. However, as we usually worried 
Early school love, poor health and often absences must affect grades but they do not really lead to a decline in grades. 

So let's visualize these findings by plotting them :)
"""

# countplot to review G3 distritution

plt.figure(figsize=(10, 6))
sns.set()
sns.countplot('G3', data=students, palette="ch:2.5,-.2,dark=.3")
plt.title('Grade distritution')
plt.xlabel('final grade')
plt.savefig(f'plots/visual/G3_countplot.png')
plt.close()
# Above plot shows that the grades of the students conform to the normal distribution.
# However there are a bit too much the students whose grade is only 0. It might be because of cheating when doing exam.
# We think the students whose grade is 0 should be removed.

# let's class our grades (high:>=15, mid:8-14, low:<=7)
high = students.loc[students['G3'] >= 15].count()[0]
medium = students.loc[(students['G3'] >= 8) & (students['G3'] <= 14)].count()[0]
low = students.loc[students['G3'] <= 7].count()[0]

# pieplot
plt.figure(figsize=(10, 6))
labels = ['high grades > = 15', 'Medium grades 8-14', 'low grades <= 7']
colors = ['#abcdef', '#aabbcc', '#67757a']
plt.pie([high, medium, low], labels=labels, colors=colors, autopct='%1.1f%%', shadow=False)
plt.title('Grades 3-classes of math and portuguese')
plt.savefig(f'plots/visual/G3_pieplot.png')
plt.close()

# lineplot on G1/G2/failures to G3
fig, ax = plt.subplots(3, 1, figsize=(10, 9))
sns.set()
index = 0
for col in ['G1', 'G2', 'failures']:
    sns.lineplot(col, 'G3', data=students, ax=ax[index])
    ax[index].set_title(col + ' to final grades')
    ax[index].set_xlabel(col)
    ax[index].set_ylabel('final Grade')
    index += 1
fig.tight_layout(pad=3.0)
plt.savefig(f'plots/visual/G3_lineplot.png')
plt.close()

# These 3 plots demonstrate that: Students' academic performance continues to be stable, which means students with
# good results will continue to perform well and vice versa.

# barplot for Medu and Fedu
new_Pstatus = []
for each_status in students['Pstatus']:
    if each_status == 0:
        new_Pstatus.append('Apart')
    else:
        new_Pstatus.append('Together')
students['NPstatus'] = new_Pstatus

plt.figure(figsize=(11, 8))
sns.set()
labels = ['Apart', 'Together']
sns.barplot('Medu', 'Fedu', hue='NPstatus', data=students, palette="Blues_d")

plt.title("Mother's education vs. Father's eduction")
plt.xlabel('Mother education level')
plt.ylabel('Father education level')
plt.xticks(np.arange(5), ('no education', 'primary(1st-4th)', 'primary(5th-9th)', 'secondary', 'university and above'))
plt.yticks(np.arange(5), ('no education', 'primary(1st-4th)', 'primary(5th-9th)', 'secondary', 'university and above'))
plt.legend()
plt.savefig(f'plots/visual/Medu_Fedu_barplot.png')
plt.close()
# This plot shows that people prefer to marry similar education background person, it might because they have more
# interests in common. And the divorce rate is almost 50% high in each group. The no education group has very small
# sample, not representative.

# Line plot
sorted_by_studytime_df = students.sort_values('studytime')
plt.figure(figsize=(12, 8))
sns.set()
sns.lineplot('studytime', 'G3', hue='sex', data=sorted_by_studytime_df)
plt.xlabel('studytime (hours/week)')
plt.ylabel('Grade')
plt.xticks([1, 2, 3, 4], ('less than 2h', '2-5hrs', '5-10hrs', 'more than 10hrs'))
plt.legend(labels=['Female', 'Male'])
plt.title('Studytime on final grade')
plt.savefig(f'plots/visual/studytime_lineplot.png')
plt.close()

# From above plot, it shows that for female students, the more studytime spent, the better the grade is. However for
# male students, the grade is increasing with the studytime, but when the total weekly studytime is over than 10hs,
# their grades are declining.

# Scatter plot
plt.style.use("bmh")
fig, axes = plt.subplots(1, 1, figsize=(10, 10))
axes.scatter(students['goout'], students['G3'], alpha=0.3, s=students[['Walc'] + ['Dalc']] ** 4,
             label='alcohol consumption')
axes.set_xlabel('Low <----going out with friends----> High')
axes.set_ylabel('Grade')
axes.set_title('Going out time on grade\n')
axes.legend()
plt.savefig(f'plots/visual/goout_scatterplot.png')
plt.close()

# From this plot we can see that the students who go out rarely has the minimal fluctuations in grades but not the
# best grade group. The best group is going out on low level but still spend sometime with friends. And all the
# groups the top grade students almost no alcohol consumption.

# swarm plot
sns.set()
plt.figure(figsize=(12, 8))
sns.swarmplot('traveltime', 'G3', data=students, hue='address', size=10)
plt.xlabel('Travel time from home to school')
plt.ylabel('Grade')
plt.xticks([0, 1, 2, 3], ('<15 min', '15-30 min', '30 min. to 1 hour', '>1 hour'))
plt.title('Travel time on grade\n')
plt.legend(labels=['Urban', 'Rural'])
plt.savefig(f'plots/visual/traveltime_swarmplot.png')
plt.close()

# From this plot we can see that the students who spent more time on the way to school have lower grades. And most of
# the students live near school and students who live in Rural are normally have more travel time than students live
# in Urban.

# comparisons

_, ax = plt.subplots(3, 2, figsize=(12, 12))
sns.set()

# First row of subplots
# Compare the percentage of extra pay on Math and Portuguese
sns.countplot('paid', data=students.loc[students['subject'] == 0], ax=ax[0][0])
ax[0][0].set_title('Extra pay on Math')
sns.countplot('paid', data=students.loc[students['subject'] == 1], ax=ax[0][1])
ax[0][1].set_title('Extra pay on Portuguese')

# Second row of subplots
# Compare Female and Male students performance on each subject
sns.boxplot('sex', 'G3', data=students.loc[students['subject'] == 0], ax=ax[1][0])
ax[1][0].set_title('G3 comparison by sex on Math')
sns.boxplot('sex', 'G3', data=students.loc[students['subject'] == 1], ax=ax[1][1])
ax[1][1].set_title('G3 comparison by sex on Portuguese')

# Third row of subplots
# Compare Mother's job and Father's job to students grade
sns.boxplot('Mjob', 'G3', data=students, ax=ax[2][0])
ax[2][0].set_title("G3 comparison by mother's job")
sns.boxplot('Fjob', 'G3', data=students, ax=ax[2][1])
ax[2][1].set_title("G3 comparison by father's job")

fig.tight_layout(pad=3.0)

plt.savefig(f'plots/visual/comparisons_plot.png')
plt.close()

# Above plots show:
# 1. For extra classes, parents paid more on Math and very little paid on Portuguese.
# 2. Male students have higher performance on Math and Female students have higher performance on Portuguese.
# 3. Mother's job is health related, their kids have best performance and Father's job is teacher related,
# their kids has best performance.

# Regplots to see other three features to final grade (Age/students willing to learn/internet availability)
fig, (axis1, axis2, axis3) = plt.subplots(1, 3, figsize=(15, 5))
sns.regplot(x='age', y='G3', data=students, ax=axis1)
sns.regplot(x='higher', y='G3', data=students, ax=axis2)
sns.regplot(x='internet', y='G3', data=students, ax=axis3)

fig.tight_layout(pad=2.0)
plt.savefig(f'plots/visual/regplots.png')
plt.close()

# Above 3 plots show:
# 1. The grade is decline with the age increasing. This is according with our normal cognition.
# The higher the grade, the knowledge is more difficult.
# 2. Students who want to take higher education get the better study performance.
# 3. Internet doesn't affect learning, it helps with academic performance.

# Data cleansing
"""
Since there are still 4 features dtype are objects, so we have to convert them to numerical data type. 
Because there is no ordinal relationship for each features, so we have to use one-hot-encoding method 
in pandas to convert categorical data to numerical data. 
"""

# use pd.concat to join the new columns with original students dataframe and drop the original 'Mjob' column
students = pd.concat([students, pd.get_dummies(students['Mjob'],
                                               prefix='Mjob', dummy_na=False)], axis=1).drop(['Mjob'], axis=1)
# use pd.concat to join the new columns with students dataframe and drop the original 'Fjob' column
students = pd.concat([students, pd.get_dummies(students['Fjob'],
                                               prefix='Fjob', dummy_na=False)], axis=1).drop(['Fjob'], axis=1)
# use pd.concat to join the new columns with students dataframe and drop the original 'reason' column
students = pd.concat([students, pd.get_dummies(students['reason'],
                                               prefix='reason', dummy_na=False)], axis=1).drop(['reason'], axis=1)
# use pd.concat to join the new columns with students dataframe and drop the original 'guardian' column
students = pd.concat([students, pd.get_dummies(students['guardian'],
                                               prefix='guardian', dummy_na=False)], axis=1).drop(['guardian'], axis=1)
# Check one-hot-encoding is applied correctly.
print(students.columns)

# need to remove 'NPstatus' - added for plotting purpose only
students.drop(['NPstatus'], axis=1, inplace=True)
print(students.shape)

# need remove students sample whose G3 is 0
students = students.loc[students['G3'] != 0]
print(students.shape)
print(students.info())  # Data cleansing is done. No data missing and all the sample dtype are numerical.

# Machine Learning - 1. Predict students final grade (Regression)
# We’re going to build up a model to estimate students final scores for two two subjects (Math and Portugues)
# from various features of the student. The scores produced are numbers between 0 and 20, where higher scores indicate
# better study performance.
def xyz(x,y):
    # Splitting features and target datasets into: train and test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35)
#    print(f"x.shape: (x.shape), y.shape: (y.shape)")
#    print(f"x_train.shape: {x_train.shape}, y_train.shape: {y_train.shape}")
#    print(f"x_test.shape: {x_test.shape}, y_test.shape: {y_test.shape}")
#    return x_train, x_test, y_train, y_test
    for Model in [LinearRegression, LinearSVR, Ridge, ElasticNet, Lasso, GradientBoostingRegressor]:
        model = Model()
        model.fit(x_train, y_train)
        predicted_values = model.predict(x_test)
        print(f"{Model}: {Model.__name__, cross_val_score(Model(), x, y).mean()}")
        print(f"MAE error(avg abs residual): {metrics.mean_absolute_error(y_test, predicted_values)}")
        print(f"MSE error: {metrics.mean_squared_error(y_test, predicted_values)}")
        print(f"RMSE error: {np.sqrt(metrics.mean_squared_error(y_test, predicted_values))}")
        print(f"R2 Score: {metrics.r2_score(y_test, predicted_values)}\n")
    return x_train, x_test, y_train, y_test
# ==> Based on the cross-validation score, we would choose "Gradient Boosting Regressor" as our predict estimator.
# Ensemble Decision Tree - Gradient Boosting Regressor
# Tuning the hyper-parameter
def gradient_booster(param_grid, n_jobs, x_train, y_train):
                      estimator = GradientBoostingRegressor()
                      classifier = GridSearchCV(estimator=estimator, cv=5, param_grid=param_grid,
                              n_jobs=n_jobs)
                      classifier.fit(x_train, y_train)
                      print(classifier.best_estimator_)
#
clf = GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
                                     learning_rate=0.05, loss='ls', max_depth=4,
                                     max_features=1.0, max_leaf_nodes=None,
                                     min_impurity_decrease=0.0, min_impurity_split=None,
                                     min_samples_leaf=3, min_samples_split=2,
                                     min_weight_fraction_leaf=0.0, n_estimators=100,
                                     n_iter_no_change=None, presort='auto',
                                     random_state=None, subsample=1.0, tol=0.0001,
                                     validation_fraction=0.1, verbose=0, warm_start=False)
def predict(x_test):
    clf.fit(x_train, y_train)

## Predicting the results for our test data set
    predicted_values = clf.predict(x_test)
#
    print(f"Printing MAE error(avg abs residual):{metrics.mean_absolute_error(y_test,predicted_values)}")
    print(f"Printing MSE error: {metrics.mean_squared_error(y_test, predicted_values)}")
    print(f"Printing RMSE error: {np.sqrt(metrics.mean_squared_error(y_test, predicted_values))}")
    print(f"R2 Score: {metrics.r2_score(y_test, predicted_values)}")
    return predicted_values

def plot_func(name, y_test, predicted_values):
## Plotting different between real and predicted values
   sns.scatterplot(y_test, predicted_values)
   plt.plot([0, 20], [0, 20], '--')
   plt.xlabel('Real Value')
   plt.ylabel('Predicted Value')
   plt.savefig(name) ###
#   plt.savefig(f'plots/ML/all_features_predict.png') ###
#   plt.savefig(f'plots/ML/withoutG2_predict.png')
#   plt.savefig(f'plots/ML/WithoutG1G2_predict.png')
   plt.close()
# Plot training deviance
test_score = np.zeros((100,), dtype=np.float64)
def plot_deviance(plot_file_name,y_test,predicted_values,x_test,test_score):
    test_score = np.zeros((100,), dtype=np.float64)

    for i, predicted_values in enumerate(clf.staged_predict(x_test)):
           test_score[i] = clf.loss_(y_test, predicted_values)
    plt.title('Deviance')
    plt.plot(np.arange(100) + 1, clf.train_score_, 'b-',
         label='Training Set Deviance')
    plt.plot(np.arange(100) + 1, test_score, 'r-',
         label='Test Set Deviance')
    plt.legend(loc='upper right')
    plt.xlabel('Boosting Iterations')
    plt.ylabel('Deviance')
    plt.savefig(plot_file_name)
   # plt.savefig(f'plots/ML/WithoutG2_deviance.png')
    #plt.savefig(f'plots/ML/WithoutG1G2_deviance.png')
    plt.close()
# Plot feature importance
def bar_func(plot_name,x):
    (pd.Series(clf.feature_importances_, index=x)
       .nlargest(10)
       .plot(kind='barh'))
    plt.title('Variable Importance')
    plt.savefig(plot_name)
    plt.close()

#################################
# ==> Based on the cross-validation score, we would choose "Gradient Boosting Regressor" as our predict estimator.

# Ensemble Decision Tree - Gradient Boosting Regressor
# Tuning the hyper-parameter

p = {'n_estimators': [100, 500],
             'learning_rate': [0.1, 0.05, 0.02],
             'max_depth': [4],
             'min_samples_leaf': [3],
             'max_features': [1.0]}

job = 4
# Data preparation (Keep all the features, including G1 and G2)
# Separating features(X) and target(y)

x1 = students.drop('G3', axis=1)
y1 = students['G3']
x_train, x_test, y_train, y_test = xyz(x1,y1)
gradient_booster(p, job, x_train, y_train)
predicted_values= predict(x_test)
plot_func(f'plots/ML/all_features_predicted_real.png', y_test, predicted_values)
plot_deviance(f'plots/ML/all_features_deviance.png',y_test,predicted_values,x_test,test_score)
bar_func(f'plots/ML/all_features_barplot.png',x1.columns)
# ==> From variable importance plotting, we can see G2 affect prediction greatly.
# Therefore, we are going to remove 'G2' from the X and see how the model performance is.

# Data preparation (Keep all the features but remove 'G2')
# Separating features(X) and target(y)

x2 = students.drop(['G3', 'G2'], axis=1)
y2 = students['G3']
x_train, x_test, y_train, y_test = xyz(x2,y2)
gradient_booster(p, job, x_train, y_train)
predicted_values= predict(x_test)
plot_func(f'plots/ML/Without_G2_predicted_real.png', y_test, predicted_values)
plot_deviance(f'plots/ML/Without_G2_deviance.png',y_test,predicted_values,x_test,test_score)
bar_func(f'plots/ML/Without_G2_barplot.png',x2.columns)

# ==> The performance of model is decline but still showing G1 has strong effect size to final grade.
# Therefore, we are going to remove  `G1 & G2` from the X and see how the model performance is.


# Data preparation - ( keep all the features without G1 ang G2)
# Separating features(X) and target(y)

x3 = students.drop(['G1', 'G2', 'G3'], axis=1)
y3 = students['G3']
x_train, x_test, y_train, y_test = xyz(x3,y3)
gradient_booster(p, job, x_train, y_train)
predicted_values= predict(x_test)
plot_func(f'plots/ML/Without_G1G2_predicted_real.png',y_test, predicted_values)
plot_deviance(f'plots/ML/Without_G1G2_deviance.png',y_test,predicted_values,x_test,test_score)
bar_func(f'plots/ML/Without_G1G2_barplot.png',x3.columns)
# ==> After removing "G1'and 'G2' grade related features, the model predictive performance is dramatically down.
# We can see all the other features are not really impact students final grade too much. In order to see more clearly,
# we are going to remove all the features only leave 'G1' and 'G2' as X.
# Therefore, we are going to check the impact of `G1` and `G2` to the model performance.

# data preparation ( keep only features: G1 and G2)
# Separating features(X) and target(y)
x4 = students[['G1', 'G2']]
y4 = students['G3']
x_train, x_test, y_train, y_test = xyz(x4,y4)
gradient_booster(p, job, x_train, y_train)
predicted_values= predict(x_test)
plot_func(f'plots/ML/OnlyG1G2_predicted_real.png', y_test, predicted_values)
plot_deviance(f'plots/ML/OnlyG1G2_deviance.png',y_test,predicted_values,x_test,test_score)
bar_func(f'plots/ML/OnlyG1G2_barplot.png',x4.columns)
"""                                          
#Summary: 

As far as the prediction model is concerned, all the characteristic variables are retained, and the prediction model 
reached is almost the same as the model that only retain the students' previous test scores (G1 and G2). If one 
previous score (G1) is removed, the accuracy of the model prediction will be reduced, but if both test scores (G1 and 
G2) are all removed, the prediction ability of the model will be greatly reduced, and it will not have the value of 
prediction. From the analysis of the different steps, we conclude: 

1. If we collect only students' previous grade (G1 and G2), we can build up a good prediction model to predict 
students' final grade. 
2. The above conclusion doesn't mean this dataset has no research value. Because from the data visualization, we get a 
lot of interesting findings. Although these findings are not always to do with academic score, they still show the value 
of social research: the differences in learning between boys and girls can provide better educational ideas for parents 
and schools. Maybe boys need more exercise than just extending study time to improve performance. And for boys and girls 
in different subjects, can schools or families provide different help. For children who do not like to socialize, they 
can properly develop their social skills, which will help improve their academic performance, but parents and school 
need to supervise problems such as alcoholism caused by improper socialization, etc. 
3. This data set does not find a characteristic variable that really affects students' academic performance. Predicting 
student next academic score with previous academic is of course the most effective and low-cost method. However, as a 
more in-depth study of the factors affecting a student's performance, this data did not find the most important 
characteristic variable. 

Suggest: Based on the above research analysis, we suggest that in data collection, it should consider the student's 
'IQ', 'EQ', or 'Expression ability', etc. These may be important factors that affect a student's academic 
performance. In the future, We hope to be able to make accurate predictions of their academic performance when there 
is no grades related information. 

"""
# Without these 3 features: grades G1 & G2 & failures, the Estimator scores are decline dramatically.
# Compare with others, Ensemble Decision Tree - 'Gradient Boosting Regressor' is still the best estimator to choose.

# Machine Learning (Fun Part)

# 2. Predict if my child has a girlfriend/boyfriend (classcification) Students in youth treason, parents sometimes
# worry very much about their children's early love. The significance of building this predictive model is to help
# parents judge whether their children have early love in the school. 0 - No, 1- Yes

# Data preparation (training data & test data)
# Separating features(X) and target(y)

X = students.drop(['romantic'], axis=1)
y = students['romantic']

# Splitting features and target datasets into: train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# Printing original Dataset
print(f"X.shape: {X.shape}, y.shape: {y.shape}")

# Printing splitted datasets
print(f"X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}")
print(f"X_test.shape: {X_test.shape}, y_test.shape: {y_test.shape}")

# Training a model using multiple differents algorithms and comparing the results
# Cross-validation to get each estimator score

for Model in [LogisticRegression, LinearSVC, neighbors.KNeighborsClassifier, SVC, GradientBoostingClassifier]:
    model = Model()
    model.fit(X_train, y_train)
    predicted_values = model.predict(X_test)
    cv = ShuffleSplit(n_splits=5)
    print(f"{Model}: {Model.__name__, cross_val_score(Model(), X, y, cv=cv)}")
    print('Classification Report')
    print(classification_report(y_test, predicted_values))

# From above cross validation score, we can see `GradientBoostingClassifier` is the best choice.
# Tuning the hyper-parameter

p5 = {'n_estimators': [100, 500],
      'learning_rate': [0.1, 0.05, 0.02],
      'max_depth': [4],
      'min_samples_leaf': [3],
      'max_features': [1.0]}

job5 = 4

gradient_booster(p5, job5,X_train,y_train)

clf = GradientBoostingClassifier(criterion='friedman_mse', init=None,
                                 learning_rate=0.1, loss='deviance', max_depth=4,
                                 max_features=1.0, max_leaf_nodes=None,
                                 min_impurity_decrease=0.0, min_impurity_split=None,
                                 min_samples_leaf=3, min_samples_split=2,
                                 min_weight_fraction_leaf=0.0, n_estimators=500,
                                 n_iter_no_change=None, presort='auto',
                                 random_state=None, subsample=1.0, tol=0.0001,
                                 validation_fraction=0.1, verbose=0,
                                 warm_start=False)

clf.fit(X_train, y_train)

# Predicting the results for our test data set
predicted_values = clf.predict(X_test)
print('Classification Report')
print(classification_report(y_test, predicted_values))
print('Confusion Matrix')
print(confusion_matrix(y_test, predicted_values))
#print('Overall f1-score')
#print(f1_score(y_test, predicted_values, average="macro"))

# plot confusion matrix
array = [[172, 29], [41, 56]]
df_cm = pd.DataFrame(array, index=[i for i in "AB"], columns=[i for i in "AB"])
plt.figure(figsize=(10, 7))
sns.heatmap(df_cm, annot=True, fmt='g')
plt.savefig(f'plots/ML/confusion_matrix.png')
plt.close()

# Although the accuracy of the model is not very high, it can somehow assist parents to prove their guess in a sense
# as an auxiliary tool.
