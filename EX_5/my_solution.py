from __future__ import print_function
from pyspark.ml.regression import LinearRegression
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.sql import functions as func

spark = SparkSession.builder.appName('DecisionTrees').getOrCreate()

data = spark.read.option('header', 'true')\
    .option('inferSchema','true')\
    .csv('file:///Users/pbednarski/Desktop/spark_course/realestate.csv')

assembler = VectorAssembler(inputCols=['HouseAge',
                                       'DistanceToMRT',
                                       'NumberConvenienceStores'],
                            outputCol='features')

df = assembler.transform(data)

trainTest = df.randomSplit([0.5, 0.5])
trainingDF = trainTest[0]
testDF = trainTest[1]

regressor = DecisionTreeRegressor(labelCol = 'PriceOfUnitArea',\
                                  featuresCol = 'features')

model = regressor.fit(trainingDF)

predictions = model.transform(testDF)

predictions.select(func.col('PriceOfUnitArea').alias('label'),\
                   func.round(func.col('prediction'), 2)\
                   .alias('prediction')).show()


