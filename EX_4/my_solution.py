from pyspark.sql import SparkSession
from pyspark.sql import functions as func
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, LongType
import sys


def computeCosineSimilarity(spark, data):
    # Compute xx, xy and yy columns
    pairScores = data \
        .withColumn("xx", func.col("rating1") * func.col("rating1")) \
        .withColumn("yy", func.col("rating2") * func.col("rating2")) \
        .withColumn("xy", func.col("rating1") * func.col("rating2"))

    # Compute numerator, denominator and numPairs columns
    calculateSimilarity = pairScores \
        .groupBy("movie1", "movie2") \
        .agg( \
        func.sum(func.col("xy")).alias("numerator"), \
        (func.sqrt(func.sum(func.col("xx"))) * func.sqrt(func.sum(func.col("yy")))).alias("denominator"), \
        func.count(func.col("xy")).alias("numPairs")
    )

    # Calculate score and select only needed columns (movie1, movie2, score, numPairs)
    result = calculateSimilarity \
        .withColumn("score", \
                    func.when(func.col("denominator") != 0, func.col("numerator") / func.col("denominator")) \
                    .otherwise(0) \
                    ).select("movie1", "movie2", "score", "numPairs")

    return result


# Get movie name by given movie id
def getMovieName(movieNames, movieId):
    result = movieNames.filter(func.col("movieID") == movieId) \
        .select("movieTitle").collect()[0]

    return result[0]


spark = SparkSession.builder.appName("MovieSimilarities").master("local[*]").getOrCreate()

movieNamesSchema = StructType([
    StructField("movieID", IntegerType(), True),
    StructField("movieTitle", StringType(), True),
    StructField("date", StringType(), True),
    StructField("none", StringType(), True),
    StructField("link", StringType(), True),
    StructField("genre1", IntegerType(), True),
    StructField("genre2", IntegerType(), True),
    StructField("genre3", IntegerType(), True),
    StructField("genre4", IntegerType(), True),
    StructField("genre5", IntegerType(), True),
    StructField("genre6", IntegerType(), True),
    StructField("genre7", IntegerType(), True),
    StructField("genre8", IntegerType(), True),
    StructField("genre9", IntegerType(), True),
    StructField("genre10", IntegerType(), True),
    StructField("genre11", IntegerType(), True),
    StructField("genre12", IntegerType(), True),
    StructField("genre13", IntegerType(), True),
    StructField("genre14", IntegerType(), True),
    StructField("genre15", IntegerType(), True),
    StructField("genre16", IntegerType(), True),
    StructField("genre17", IntegerType(), True),
    StructField("genre18", IntegerType(), True),
    StructField("genre19", IntegerType(), True)
])

moviesSchema = StructType([ \
    StructField("userID", IntegerType(), True), \
    StructField("movieID", IntegerType(), True), \
    StructField("rating", IntegerType(), True), \
    StructField("timestamp", LongType(), True)])

# Create a broadcast dataset of movieID and movieTitle.
# Apply ISO-885901 charset
movieNames = spark.read \
    .option("sep", "|") \
    .option("charset", "ISO-8859-1") \
    .schema(movieNamesSchema) \
    .csv("file:///Users/pbednarski/Desktop/spark_course/ml-100k/u.item")

movieNamesData = movieNames.select('movieID',
                                   'movieTitle',
                                   'genre1',
                                   'genre2',
                                   'genre3',
                                   'genre4',
                                   'genre5',
                                   'genre6',
                                   'genre7',
                                   'genre8',
                                   'genre9',
                                   'genre10',
                                   'genre11',
                                   'genre12',
                                   'genre13',
                                   'genre14',
                                   'genre15',
                                   'genre16',
                                   'genre17',
                                   'genre18',
                                   'genre19')

# Load up movie data as dataset
movies = spark.read \
    .option("sep", "\t") \
    .schema(moviesSchema) \
    .csv("file:///Users/pbednarski/Desktop/spark_course/ml-100k/u.data")

ratings = movies.select("userId", "movieId", "rating")

# Emit every movie rated together by the same user.
# Self-join to find every combination.
# Select movie pairs and rating pairs
moviePairs = ratings.alias("ratings1") \
    .join(ratings.alias("ratings2"), (func.col("ratings1.userId") == func.col("ratings2.userId")) \
          & (func.col("ratings1.movieId") < func.col("ratings2.movieId"))) \
    .select(func.col("ratings1.movieId").alias("movie1"), \
            func.col("ratings2.movieId").alias("movie2"), \
            func.col("ratings1.rating").alias("rating1"), \
            func.col("ratings2.rating").alias("rating2"))

goodMoviePairs = moviePairs.filter((func.col('rating1') > 3) & (func.col('rating2') > 3))

moviePairSimilarities = computeCosineSimilarity(spark, goodMoviePairs)

# get all genres column names
column_names = movieNamesData.columns
column_names = [name for name in column_names if name not in ['movieID', 'movieTitle']]

# concatenating genres columns to one list column
movieNamesData = movieNamesData.withColumn('genres', func.concat(*column_names))

movieGenressData = movieNamesData.select('movieID', 'genres')

# joining genres1 column for movie1 and genres2 column for movie2

joinedMoviePairSimilarities = moviePairSimilarities.join(
    movieGenressData,
    func.col("movie1") == func.col("movieID")
).drop('movieID')

joinedMoviePairSimilarities = joinedMoviePairSimilarities.withColumnRenamed('genres', 'genres1')

joinedMoviePairSimilarities = joinedMoviePairSimilarities.join(
    movieGenressData,
    func.col("movie2") == func.col("movieID")
).drop('movieID')

joinedMoviePairSimilarities = joinedMoviePairSimilarities.withColumnRenamed('genres', 'genres2')


def compare(genre1, genre2):
    matching_ones = sum(1 for x, y in zip(genre1, genre2) if x == y == '1')
    return matching_ones


compare_udf = func.udf(compare, IntegerType())

# Calculating genreScore for every pair of films with udf function
joinedSimilaritiesWGenre = joinedMoviePairSimilarities.withColumn('genreScored', \
                                                                  compare_udf(joinedMoviePairSimilarities['genres1'], \
                                                                              joinedMoviePairSimilarities['genres2']))

finalTable = joinedSimilaritiesWGenre.select('movie1', 'movie2', 'score', \
                                             'numPairs', 'genreScored')

if (len(sys.argv) > 1):
    scoreThreshold = 0.97
    coOccurrenceThreshold = 100.0
    genreTreshold = 2

    movieID = int(sys.argv[1])

    # Filter for movies with this sim that are "good" as defined by
    # our quality thresholds above
    filteredResults = finalTable.filter( \
        ((func.col("movie1") == movieID) | (func.col("movie2") == movieID)) & \
        (func.col("score") > scoreThreshold) & (func.col("numPairs") > coOccurrenceThreshold) \
        & (func.col("genreScored") > func.lit(genreTreshold)))

    null_value = 0
    next_result = filteredResults.where((func.col("genreScored") > func.lit(null_value)))

    # Sort by quality score.
    results = filteredResults.sort(func.col("score").desc()).take(10)

    print("Top 10 similar movies for " + getMovieName(movieNamesData, movieID))

    for result in results:
        # Display the similarity result that isn't the movie we're looking at
        similarMovieID = result.movie1
        if (similarMovieID == movieID):
            similarMovieID = result.movie2

        print(getMovieName(movieNamesData, similarMovieID) + "\tscore: " \
              + str(result.score) + "\tstrength: " + str(result.numPairs) \
              + "\tgenre similarity: " + str(result.genreScored))


