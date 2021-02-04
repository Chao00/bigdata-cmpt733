import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+

from pyspark.sql import SparkSession, functions, types
spark = SparkSession.builder.appName('tmax model tester').getOrCreate()
assert spark.version >= '2.3' # make sure we have Spark 2.3+
spark.sparkContext.setLogLevel('WARN')

from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import elevation_grid as eg

# Note: This is my schema for b1 please use this for b1
my_tmax_schema = types.StructType([
    types.StructField('latitude', types.FloatType()),
    types.StructField('longitude', types.FloatType()),
    types.StructField('elevation', types.FloatType()),
    types.StructField('date', types.DateType()),
    types.StructField('tmax', types.FloatType()),
    types.StructField('station', types.StringType()) 
])

#This is the schema for tmax-test, use this when apply my modle to tmax-test 
# tmax_schema = types.StructType([
#     types.StructField('station', types.StringType()),
#     types.StructField('date', types.DateType()),
#     types.StructField('latitude', types.FloatType()),
#     types.StructField('longitude', types.FloatType()),
#     types.StructField('elevation', types.FloatType()),
#     types.StructField('tmax', types.FloatType()),
# ])  


def test_model(model_file, inputs, output):
    # get the data
    test_tmax = spark.read.csv(inputs, schema=my_tmax_schema)

    # load the model
    model = PipelineModel.load(model_file)
    
    # use the model to make predictions
    predictions = model.transform(test_tmax)
    predictions.show()
    # save the prediction as csv file for plot
    df = predictions.toPandas()
    df.to_csv(output + ".csv", encoding='utf-8')
    
    # evaluate the predictions
    r2_evaluator = RegressionEvaluator(predictionCol='prediction', labelCol='tmax',
            metricName='r2')
    r2 = r2_evaluator.evaluate(predictions)
    
    rmse_evaluator = RegressionEvaluator(predictionCol='prediction', labelCol='tmax',
            metricName='rmse')
    rmse = rmse_evaluator.evaluate(predictions)

    print('r2 =', r2)
    print('rmse =', rmse)

if __name__ == '__main__':
    model_file = sys.argv[1]
    inputs = sys.argv[2]
    output = sys.argv[3]
    test_model(model_file, inputs, output)
