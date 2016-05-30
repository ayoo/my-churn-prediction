from sklearn import metrics
from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.feature import *
from pyspark.sql.types import *
from pyspark.sql.functions import udf
from pyspark.mllib.linalg import  VectorUDT

def get_schema():
    schema = StructType((
        StructField("nth_membership_period", DoubleType(), True),
        StructField("Window", DoubleType(), True),
        StructField("Usage_Days", DoubleType(), True),
        StructField("Supercategory_Ratio",DoubleType(), True),
        StructField("Intensity_Ratio", DoubleType(), True),
        StructField("Product_Article", DoubleType(), True),
        StructField("Overview_Detail", DoubleType(), True),
        StructField("Email_Open_R", DoubleType(), True),
        StructField("Email_Clicked_R", DoubleType(), True),
        StructField("Browser_Researcher", DoubleType(), True),
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

def safe_float(str):
    try:
        return float(str)
    except StandardError as se:
        #print(se)
        return 0.0

def read_dataset(sc, filename):
    sql_context = SQLContext(sc)
    rdd = sc.textFile(filename)\
        .map(lambda line: line.split(",")) \
        .map(lambda r: [safe_float(r[4]), safe_float(r[20]), safe_float(r[17]), safe_float(r[80]), safe_float(r[77]), safe_float(r[81]),
                        safe_float(r[82]),safe_float(r[86]), safe_float(r[87]), safe_float(r[88]), 'Renewers' if r[78] == 'Renewers' else 'Others'])

    df = sql_context.createDataFrame(rdd, get_schema())
    #newdf = df.na.fill(0.0)
    #print 'Null count >>>>>', newdf.where(newdf['Product_Article'].isNull()).count()
    return df

def pipe_index_string_cols(df, cols=[]):
    newdf = df
    for col in cols:
        si = StringIndexer(inputCol=col, outputCol=col+"-idx")
        model = si.fit(newdf)
        newdf = model.transform(df).drop(col).withColumnRenamed(col+"-idx", col)
    return newdf

def pipe_one_hot_encode_cols(df, cols=[], dropLast=True):
    newdf = df
    for col in cols:
        oh = OneHotEncoder(inputCol=col, outputCol=col+"-ohe", dropLast=dropLast)
        model = oh.fit(newdf)
        newdf = model.transform(df).drop(col).withColumnRenamed(col+"-ohe", col)
    return newdf

def pipe_assemble_features(df, excluded_cols=[]):
    features = list(set(df.columns) - set(excluded_cols))
    print features
    va = VectorAssembler(inputCols=features, outputCol="features")
    newdf = va.transform(df)
    return newdf

def pipe_scale_cols(df, with_mean=True, with_std=True, use_dense_vector=True):
    newdf = df
    if use_dense_vector:
        to_dense_udf = udf(lambda v: v.toDense, VectorUDT())
        dense_df = newdf.withColumn("features-dense", to_dense_udf(newdf["features"]))
        newdf = dense_df.drop("features").withColumnRenamed("features-dense", "features")

    scaler = StandardScaler(withMean=with_mean, withStd=False, inputCol="features", outputCol="features-scaled")
    model = scaler.fit(newdf)
    newdf = model.transform(newdf)
    newdf = newdf.drop("features")
    newdf = newdf.withColumnRenamed("features-scaled", "features")
    return newdf

def train_test_split(df, test_rate):
    (training_data, test_data) = df.randomSplit([1 - test_rate, test_rate])
    training_data.cache()
    test_data.cache()
    return training_data, test_data

def rf_classifier():
    classifier =  RandomForestClassifier(labelCol="label", numTrees=5, maxBins=30, maxDepth=10, impurity='gini')
    return classifier

def evaluate(predictions, spark_metrics):
    # using sklearn metrics
    y_hat  = predictions.rdd.map(lambda p: p.prediction).collect()
    y_true = predictions.rdd.map(lambda p: p.label).collect()

    print metrics.classification_report(y_true, y_hat)
    print 'AUC score: %f' %  metrics.roc_auc_score(y_true, y_hat)
    print("Accuracy: %f" % metrics.accuracy_score(y_true, y_hat))

    # using spark metrics
    result = []
    for metric in spark_metrics:
        eval = BinaryClassificationEvaluator().setMetricName(metric)
        result.append(eval.evaluate(predictions))
    return result

def main():
    conf = SparkConf()
    conf.setMaster('local[*]')
    conf.setAppName('renewer-prediction-spark')
    filename = '/Users/andyyoo/scikit_learn_data/renewer/Orange_Dataset.no.header.csv'
    sc = SparkContext(conf=conf)
    df = read_dataset(sc, filename)
    df = pipe_index_string_cols(df, cols=["label"])
    df = pipe_assemble_features(df, excluded_cols=["label"])
    df = pipe_scale_cols(df, with_mean=True, with_std=True, use_dense_vector=False)
    df.show()

    training_data, test_data = train_test_split(df, 0.2)
    model = rf_classifier().fit(training_data)
    predictions = model.transform(test_data)
    print predictions.show(20)

    (roc_score, pr_score) = evaluate(predictions, ['areaUnderROC', 'areaUnderPR'])
    print "\nSpark AUC Score: ", roc_score, ", PR Score: ", pr_score


if __name__ == '__main__':
    main()