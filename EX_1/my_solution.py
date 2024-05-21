from pyspark import SparkConf, SparkContext

conf = SparkConf().setMaster('local').setAppName('CustomerCount')
sc = SparkContext(conf = conf)

def parse_lines(line):
    fields = line.split(',')
    customer_id = int(fields[0])
    price_paid = float(fields[2])
    return customer_id, price_paid

input = sc.textFile('file://customer-orders.csv') 
parsedLines = input.map(parse_lines)
customers_total = parsedLines.reduceByKey(lambda x, y: x + y)
customers_total = customers_total.map(lambda x: (x[1], x[0])).sortByKey()
results = customers_total.collect()

for result in results:
    print(result[0], '  ', result[1])