import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+

from pyspark.sql import SparkSession, functions, types
spark = SparkSession.builder.appName('colour prediction').getOrCreate()
spark.sparkContext.setLogLevel('WARN')
assert spark.version >= '2.4' # make sure we have Spark 2.4+

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, SQLTransformer
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import AFTSurvivalRegression, DecisionTreeRegressor, GBTRegressor

tmax_schema = types.StructType([
    types.StructField('station', types.StringType()),
    types.StructField('date', types.DateType()),
    types.StructField('latitude', types.FloatType()),
    types.StructField('longitude', types.FloatType()),
    types.StructField('elevation', types.FloatType()),
    types.StructField('tmax', types.FloatType()),
])  

def main(inputs):
    data = spark.read.csv(inputs, schema=tmax_schema)
    train, validation = data.randomSplit([0.75, 0.25])
    train = train.cache()
    validation = validation.cache()
    data.coalesce(1).write.format("csv").save("weather-2")
    date_transformer = SQLTransformer(statement= "SELECT *, DAYOFYEAR(date) AS dayOfYear FROM  __THIS__")
    statement = "SELECT latitude, longitude, elevation, tmax, dayOfYear FROM __THIS__"
    data_transformer = SQLTransformer(statement= statement)
    weather_assembler = VectorAssembler(inputCols = ['latitude', 'longitude', 'elevation', 'dayOfYear'], outputCol="features")

    GBTR = GBTRegressor(featuresCol = 'features', labelCol = 'tmax')
    weather_assembler_pipeline = Pipeline(stages=[date_transformer, data_transformer, weather_assembler, GBTR])

    weather_model = weather_assembler_pipeline.fit(train)

    predictions = weather_model.transform(validation)
    predictions.show()

    r2_evaluator = RegressionEvaluator(predictionCol = "prediction", labelCol = "tmax", metricName = 'r2')
    r2 = r2_evaluator.evaluate(predictions)

    rmse_evaluator = RegressionEvaluator(predictionCol='prediction', labelCol='tmax',
            metricName='rmse')
    rmse = rmse_evaluator.evaluate(predictions)
    print('r2 =', r2)
    print('rmse =', rmse)
    weather_model.write().overwrite().save(model_file)
    
if __name__ == '__main__':
    inputs = sys.argv[1]
    model_file = sys.argv[2]
    main(inputs)



