from pyspark.sql import SparkSession
from pyspark.sql import functions as func

spark = SparkSession.builder.appName('SparkSQL_EX').getOrCreate()

people = spark.read.option('header', 'true').option('inferSchema', 'true')\
    .csv('/Users/pbednarski/Desktop/spark_course/fakefriends-header.csv')

people_info = people.select('age', 'friends')
people_info.groupBy('age').agg(func.round(func.avg('friends'), 2).alias('friends_avg')).sort('age').show()

spark.stop()