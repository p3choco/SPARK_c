from pyspark.sql import SparkSession
from pyspark.sql import functions as func
from pyspark.sql.types import IntegerType, FloatType, StructType, StructField, StringType

spark = SparkSession.builder.appName('CustomerOrdersCount').getOrCreate()

schema = StructType([\
                    StructField('customerID', IntegerType(), True), \
                    StructField('productID', StringType(), True), \
                    StructField('productPrice', FloatType(), True)])

df = spark.read.schema(schema).csv('file:///Users/pbednarski/Desktop/spark_course/customer-orders.csv')
df.printSchema()

customer_grupped = df.groupBy('customerID').agg(func.round(func.sum('productPrice'),2)
                    .alias('price_per_customer')).sort('price_per_customer')
customer_grupped.show()

spark.stop()