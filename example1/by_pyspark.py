from sklearn import metrics

from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.feature import *
from pyspark.sql.types import *


def get_churn_schema():
    schema = StructType((
        StructField("state", StringType(), True),
        StructField("account_length", DoubleType(), True),
        StructField("area_code",DoubleType(), True),
        StructField("phone", StringType(), True),
        StructField("int_plan", BooleanType(), True),
        StructField("vmail_plan", BooleanType(), True),
        StructField("vmail_message", DoubleType(), True),
        StructField("day_mins", DoubleType(), True),
        StructField("day_calls", DoubleType(), True),
        StructField("day_charge", DoubleType(), True),
        StructField("eve_mins", DoubleType(), True),
        StructField("eve_calls", DoubleType(), True),
        StructField("eve_charge", DoubleType(), True),
        StructField("night_mins", DoubleType(), True),
        StructField("night_calls", DoubleType(), True),
        StructField("night_charge", DoubleType(), True),
        StructField("int_mins", DoubleType(), True),
        StructField("int_calls", DoubleType(), True),
        StructField("int_charge", DoubleType(), True),
        StructField("custserve_calls", DoubleType(), True),
        StructField("label", StringType(), True)))
    return schema

def reformat_element(element):
    try:
        if element == 'yes':
            return True
        elif element == 'no':
            return False
        else:
            return float(element)
    except ValueError as ve:
        return element

def read_dataset(sc, filename):
    sql_context = SQLContext(sc)
    drop_col_index = (0,2,3)
    csv_data = sc.textFile(filename).map(lambda line: line.split(","))
    chrun_rdd = csv_data.map(lambda line: [reformat_element(e) for e in line])
    churn_df = sql_context.createDataFrame(chrun_rdd, get_churn_schema())

    features = list(set(churn_df.columns) - set(["state","area_code","phone","label"]))
    va = VectorAssembler().setOutputCol("features")
    va.setInputCols(features)
    return va.transform(churn_df).select("features", "label")

def build_pipeline():
    classifier = rf_classifier()

    label_indexer = StringIndexer()\
        .setInputCol("label")\
        .setOutputCol("indexedLabel")

    feature_indexer = VectorIndexer()\
        .setInputCol("features")\
        .setOutputCol("indexedFeatures")\
        .setMaxCategories(10)

    label_converter = IndexToString()\
        .setInputCol("prediction")\
        .setOutputCol("predictedLabel")\
        .setLabels(['False.','True.'])

    stages = [label_indexer, feature_indexer, classifier, label_converter]
    pipeline = Pipeline().setStages(stages)
    return pipeline

def train_test_split(churn_df, test_rate):
    (training_data, test_data) = churn_df.randomSplit([1 - test_rate, test_rate])
    training_data.cache()
    test_data.cache()
    return training_data, test_data

def rf_classifier():
    classifier =  RandomForestClassifier().\
        setLabelCol("indexedLabel").\
        setFeaturesCol("indexedFeatures").\
        setImpurity("gini").\
        setMaxBins(30).\
        setMaxDepth(10).\
        setNumTrees(15)
    return classifier

def evaluate(predictions, spark_metrics):
    # using sklearn metrics
    y_hat  = predictions.rdd.map(lambda p: p.prediction).collect()
    y_true = predictions.rdd.map(lambda p: p.indexedLabel).collect()

    print metrics.classification_report(y_true, y_hat)
    print 'AUC score: %f' %  metrics.roc_auc_score(y_true, y_hat)
    print("Accuracy: %f" % metrics.accuracy_score(y_true, y_hat))

    # using spark metrics
    result = []
    for metric in spark_metrics:
        eval = BinaryClassificationEvaluator()\
            .setLabelCol("indexedLabel")\
            .setMetricName(metric)
        result.append(eval.evaluate(predictions))
    return result

def main():
    conf = SparkConf()
    conf.setMaster('local[*]')
    conf.setAppName('spark-basic')
    sc = SparkContext(conf=conf)
    churn_df = read_dataset(sc, "churn_no_header.csv")
    pipeline = build_pipeline()
    training_data, test_data = train_test_split(churn_df, 0.2)
    model = pipeline.fit(training_data)
    predictions = model.transform(test_data)
    print predictions.show(20)

    (roc_score, pr_score) = evaluate(predictions, ['areaUnderROC', 'areaUnderPR'])
    print "\nSpark AUC Score: ", roc_score, ", PR Score: ", pr_score

if __name__ == '__main__':
    main()

'''
Best Predictions
             precision    recall  f1-score   support

        0.0       0.95      0.99      0.97       561
        1.0       0.93      0.74      0.83       105

avg / total       0.95      0.95      0.95       666

AUC score: 0.866081
Accuracy: 0.950450

Spark AUC Score:  0.92143281555 , PR Score:  0.85252627913

# Not sure why AUC are different between sklearn metrics and Spark merics. Needs some investigation.
'''