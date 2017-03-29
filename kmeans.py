from pyspark.mllib.clustering import KMeans, KMeansModel
from pyspark import SparkContext, SparkConf
from math import sqrt
import logging, sys

#spark-submit --py-files master/hadoop/lda.py --packages com.databricks:spark-csv_2.10:1.4.0 --master yarn --deploy-mode cluster --driver-memory 50g --num-executors 30 master/hadoop/kmeans.py

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

sc = SparkContext(appName='kmeans_w2v')
from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)
data = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").load("w2v_vector.csv")
from pyspark.mllib.linalg import Vectors
data = data.map(lambda x: [float(a) for a in x])
df = data.toDF()
columns = df.columns
vectors = df.select(columns[1:71])
vectors_rdd = vectors.rdd
vectors_rdd = vectors_rdd.map(lambda x: Vectors.dense(x))

clusters = None

sqlContext = SQLContext(sc)
wssses = []
for n_clusters in range(10,500,20):
    # Build the model (cluster the data)
    clusters = KMeans.train(data, n_clusters, maxIterations=10, runs=10, initializationMode="random")

    # Evaluate clustering by computing Within Set Sum of Squared Errors
    def error(point):
        center = clusters.centers[clusters.predict(point)]
        # http://stackoverflow.com/questions/32977641/index-out-of-range-in-spark-mllib-k-means-with-tfidf-for-text-clutsering
        return sqrt(sum([x**2 for x in (point - center)]))
        # return sqrt(sum([x ** 2 for x in (point.toArray() - center)]))


    WSSSE = data.map(lambda point: error(point)).reduce(lambda x, y: x + y)
    logger.info("Within Set Sum of Squared Error = " + str(n_clusters) + "&" +  str(WSSSE) + "\\")
    wssses.append([n_clusters, WSSSE])

# Save and load model
clusters.save(sc, "w2v_clusters")

wssse_df = sqlContext.createDataFrame(wssses, ["n_clusters", "WSSSE"])
wssse_df.save("wssse")