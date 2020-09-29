#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 14:21:21 2020

@author: mariamawitanteneh
"""

# Read Text Data
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('text mining').getOrCreate()
data = spark.read.csv("/Users/mariamawitanteneh/Desktop/IOT/farm-ads.csv", inferSchema=True)
data = data.withColumnRenamed('_c0','class').withColumnRenamed('_c1','text')
data.show(20)
"""
+-----+--------------------+
|class|                text|
+-----+--------------------+
|    1| ad-jerry ad-bruc...|
|   -1| ad-rheumatoid ad...|
|   -1| ad-rheumatologis...|
|   -1| ad-siemen ad-wat...|
|   -1| ad-symptom ad-mu...|
|    1| ad-animal ad-ani...|
|   -1| ad-dr ad-enrico ...|
|   -1| ad-ulcerative ad...|
|   -1| ad-wellcentive a...|
|    1| ad-free ad-raw a...|
|   -1| ad-north ad-shor...|
|    1| ad-world ad-fine...|
|    1| ad-vet ad-online...|
|   -1| ad-gum ad-diseas...|
|    1| ad-rabbit ad-gui...|
|   -1| ad-colitis ad-sy...|
|   -1| ad-disease ad-si...|
|    1| ad-pygmy ad-goat...|
|    1| ad-feed ad-suppl...|
|   -1| ad-www ad-muscle...|
+-----+--------------------+
only showing top 20 rows
"""

# In[2]:
# Count number of Words in each Text
from pyspark.sql.functions import length
data = data.withColumn('length', length(data['text']))
data.show()
"""
+-----+--------------------+------+
|class|                text|length|
+-----+--------------------+------+
|    1| ad-jerry ad-bruc...|   101|
|   -1| ad-rheumatoid ad...|  3817|
|   -1| ad-rheumatologis...|   657|
|   -1| ad-siemen ad-wat...|  1317|
|   -1| ad-symptom ad-mu...|   107|
|    1| ad-animal ad-ani...|    79|
|   -1| ad-dr ad-enrico ...|   159|
|   -1| ad-ulcerative ad...|  2867|
|   -1| ad-wellcentive a...|  1631|
|    1| ad-free ad-raw a...|  1039|
|   -1| ad-north ad-shor...|   847|
|    1| ad-world ad-fine...|  1524|
|    1| ad-vet ad-online...|  6841|
|   -1| ad-gum ad-diseas...|  7522|
|    1| ad-rabbit ad-gui...| 47351|
|   -1| ad-colitis ad-sy...|  8939|
|   -1| ad-disease ad-si...|  1419|
|    1| ad-pygmy ad-goat...|  3422|
|    1| ad-feed ad-suppl...|  1559|
|   -1| ad-www ad-muscle...| 14759|
+-----+--------------------+------+
only showing top 20 rows
"""

# In[3]:
# Compare the length difference 
data.groupby('class').mean().show()

"""
+-----+----------+------------------+
|class|avg(class)|       avg(length)|
+-----+----------+------------------+
|   -1|      -1.0|2919.4418003103983|
|    1|       1.0| 3484.166968325792|
+-----+----------+------------------+
"""

# In[4]:
# Treat TF-IDF features for each text
# TF: Term Frequency
# IDF: Inverse Document Frequency
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF, StringIndexer, VectorAssembler

tokenizer = Tokenizer(inputCol="text", outputCol="token_text")
stopremove = StopWordsRemover(inputCol='token_text',outputCol='stop_tokens')
count_vec = CountVectorizer(inputCol='stop_tokens',outputCol='c_vec')
idf = IDF(inputCol="c_vec", outputCol="tf_idf")
farm_ads_to_num = StringIndexer(inputCol='class',outputCol='label')
final_feature = VectorAssembler(inputCols=['tf_idf', 'length'],outputCol='features')

from pyspark.ml import Pipeline
data_prep_pipe = Pipeline(stages=[farm_ads_to_num,tokenizer,stopremove,count_vec,idf,final_feature])
clean_data = data_prep_pipe.fit(data).transform(data)

clean_data.show()
clean_data.take(1)
clean_data.take(1)[0][-1]

"""
+-----+--------------------+------+-----+--------------------+--------------------+--------------------+--------------------+--------------------+
|class|                text|length|label|          token_text|         stop_tokens|               c_vec|              tf_idf|            features|
+-----+--------------------+------+-----+--------------------+--------------------+--------------------+--------------------+--------------------+
|    1| ad-jerry ad-bruc...|   101|  0.0|[, ad-jerry, ad-b...|[, ad-jerry, ad-b...|(54858,[36,55,136...|(54858,[36,55,136...|(54859,[55,136,11...|
|   -1| ad-rheumatoid ad...|  3817|  1.0|[, ad-rheumatoid,...|[, ad-rheumatoid,...|(54858,[0,2,3,6,7...|(54858,[0,2,3,6,7...|(54859,[0,2,3,6,7...|
|   -1| ad-rheumatologis...|   657|  1.0|[, ad-rheumatolog...|[, ad-rheumatolog...|(54858,[3,7,36,61...|(54858,[3,7,36,61...|(54859,[3,7,61,65...|
|   -1| ad-siemen ad-wat...|  1317|  1.0|[, ad-siemen, ad-...|[, ad-siemen, ad-...|(54858,[1,24,29,3...|(54858,[1,24,29,3...|(54859,[1,24,29,3...|
|   -1| ad-symptom ad-mu...|   107|  1.0|[, ad-symptom, ad...|[, ad-symptom, ad...|(54858,[36,55,136...|(54858,[36,55,136...|(54859,[55,136,81...|
|    1| ad-animal ad-ani...|    79|  0.0|[, ad-animal, ad-...|[, ad-animal, ad-...|(54858,[36,55,136...|(54858,[36,55,136...|(54859,[55,136,30...|
|   -1| ad-dr ad-enrico ...|   159|  1.0|[, ad-dr, ad-enri...|[, ad-dr, ad-enri...|(54858,[36,55,136...|(54858,[36,55,136...|(54859,[55,136,25...|
|   -1| ad-ulcerative ad...|  2867|  1.0|[, ad-ulcerative,...|[, ad-ulcerative,...|(54858,[1,3,4,10,...|(54858,[1,3,4,10,...|(54859,[1,3,4,10,...|
|   -1| ad-wellcentive a...|  1631|  1.0|[, ad-wellcentive...|[, ad-wellcentive...|(54858,[3,5,7,12,...|(54858,[3,5,7,12,...|(54859,[3,5,7,12,...|
|    1| ad-free ad-raw a...|  1039|  0.0|[, ad-free, ad-ra...|[, ad-free, ad-ra...|(54858,[1,2,3,12,...|(54858,[1,2,3,12,...|(54859,[1,2,3,12,...|
|   -1| ad-north ad-shor...|   847|  1.0|[, ad-north, ad-s...|[, ad-north, ad-s...|(54858,[3,6,7,18,...|(54858,[3,6,7,18,...|(54859,[3,6,7,18,...|
|    1| ad-world ad-fine...|  1524|  0.0|[, ad-world, ad-f...|[, ad-world, ad-f...|(54858,[2,7,24,32...|(54858,[2,7,24,32...|(54859,[2,7,24,32...|
|    1| ad-vet ad-online...|  6841|  0.0|[, ad-vet, ad-onl...|[, ad-vet, ad-onl...|(54858,[0,2,3,7,8...|(54858,[0,2,3,7,8...|(54859,[0,2,3,7,8...|
|   -1| ad-gum ad-diseas...|  7522|  1.0|[, ad-gum, ad-dis...|[, ad-gum, ad-dis...|(54858,[1,3,4,5,7...|(54858,[1,3,4,5,7...|(54859,[1,3,4,5,7...|
|    1| ad-rabbit ad-gui...| 47351|  0.0|[, ad-rabbit, ad-...|[, ad-rabbit, ad-...|(54858,[0,1,2,3,4...|(54858,[0,1,2,3,4...|(54859,[0,1,2,3,4...|
|   -1| ad-colitis ad-sy...|  8939|  1.0|[, ad-colitis, ad...|[, ad-colitis, ad...|(54858,[1,2,3,5,7...|(54858,[1,2,3,5,7...|(54859,[1,2,3,5,7...|
|   -1| ad-disease ad-si...|  1419|  1.0|[, ad-disease, ad...|[, ad-disease, ad...|(54858,[2,3,13,17...|(54858,[2,3,13,17...|(54859,[2,3,13,17...|
|    1| ad-pygmy ad-goat...|  3422|  0.0|[, ad-pygmy, ad-g...|[, ad-pygmy, ad-g...|(54858,[3,5,7,8,1...|(54858,[3,5,7,8,1...|(54859,[3,5,7,8,1...|
|    1| ad-feed ad-suppl...|  1559|  0.0|[, ad-feed, ad-su...|[, ad-feed, ad-su...|(54858,[1,3,7,13,...|(54858,[1,3,7,13,...|(54859,[1,3,7,13,...|
|   -1| ad-www ad-muscle...| 14759|  1.0|[, ad-www, ad-mus...|[, ad-www, ad-mus...|(54858,[0,1,2,3,5...|(54858,[0,1,2,3,5...|(54859,[0,1,2,3,5...|
+-----+--------------------+------+-----+--------------------+--------------------+--------------------+--------------------+--------------------+
only showing top 20 rows


Out[90]: [Row(class=1, text=' ad-jerry ad-bruckheimer ad-chase ad-premier ad-sept ad-th ad-clip ad-bruckheimer 
ad-chase page found', length=101, label=0.0, token_text=['', 'ad-jerry', 'ad-bruckheimer', 'ad-chase', 
'ad-premier', 'ad-sept', 'ad-th', 'ad-clip', 'ad-bruckheimer', 'ad-chase', 'page', 'found'], 
stop_tokens=['', 'ad-jerry', 'ad-bruckheimer', 'ad-chase', 'ad-premier', 'ad-sept', 'ad-th', 
'ad-clip', 'ad-bruckheimer', 'ad-chase', 'page', 'found'], c_vec=SparseVector(54858, 
{36: 1.0, 55: 1.0, 136: 1.0, 11429: 1.0, 12621: 1.0, 15066: 1.0, 18025: 2.0, 26956: 1.0, 
32013: 2.0, 44975: 1.0}), tf_idf=SparseVector(54858, {36: 0.0, 55: 0.9473, 136: 1.2784, 
11429: 5.9315, 12621: 6.25, 15066: 6.5377, 18025: 13.8862, 26956: 7.2308, 32013: 15.2725, 
44975: 7.6363}), features=SparseVector(54859, {55: 0.9473, 136: 1.2784, 11429: 5.9315, 
12621: 6.25, 15066: 6.5377, 18025: 13.8862, 26956: 7.2308, 32013: 15.2725, 44975: 7.6363, 54858: 101.0}))]



Out[76]: SparseVector(54859, {55: 0.9473, 136: 1.2784, 11534: 5.9315, 13054: 6.25, 15596: 6.5377, 20263: 
    13.8862, 30635: 15.2725, 32481: 7.2308, 53869: 7.6363, 54858: 101.0})

    
--ANSWER    
i) The matrix is a sparse matrix because from the result above we can see of the 54859, the ones shown in 
the output are non-zeros. This means that most of the other elements are 0s indicating that it is a sparse
matrix. It is also indicated in the output by SparseVector chhanging it to an array will show most the
elements being 0. 

ii) We can take the second and third rows from the result above for non-zero entries. The non-zeros have a
class lable of -1 which means it is an irrelevant ad. the text token shows the words separated by comma
and from the c_vec we can see all the words in the file and then the word count for each word in a row
as separated by comma in the text column. The wee convert tf_idf which is multiplied to the term frequency.
features is length added to it. 

"""


# In[5]: 
# ## Split data into training and test datasets
training, test = clean_data.randomSplit([0.6, 0.4], seed=12345)

# Build Logistic Regression Model
from pyspark.ml.classification import LogisticRegression

log_reg = LogisticRegression(featuresCol='features', labelCol='label')
logr_model = log_reg.fit(training)

results = logr_model.transform(test)
results.select('label','prediction').show()

"""
+-----+----------+
|label|prediction|
+-----+----------+
|  1.0|       1.0|
|  1.0|       1.0|
|  1.0|       1.0|
|  1.0|       1.0|
|  1.0|       1.0|
|  1.0|       1.0|
|  1.0|       1.0|
|  1.0|       1.0|
|  1.0|       1.0|
|  1.0|       1.0|
|  1.0|       1.0|
|  1.0|       1.0|
|  1.0|       1.0|
|  1.0|       1.0|
|  1.0|       1.0|
|  1.0|       1.0|
|  1.0|       1.0|
|  1.0|       1.0|
|  1.0|       1.0|
|  1.0|       1.0|
+-----+----------+
only showing top 20 rows
"""


# In[10]:
# #### Confusion Matrix
from sklearn.metrics import confusion_matrix
y_true = results.select("label")
y_true = y_true.toPandas()

y_pred = results.select("prediction")
y_pred = y_pred.toPandas()

cnf_matrix = confusion_matrix(y_true, y_pred)
print(cnf_matrix)
print("Prediction Accuracy is ", (cnf_matrix[0,0]+cnf_matrix[1,1])/sum(sum(cnf_matrix)) )

"""
[[783 108]
 [148 623]]
Prediction Accuracy is  0.8459687123947052
"""