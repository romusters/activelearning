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
def train():
    sqlContext = SQLContext(sc)
    data = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").load("w2v_vector.csv")
    from pyspark.mllib.linalg import Vectors
    data = data.map(lambda x: [float(a) for a in x])

    df = data.toDF()
    df = df.sample(False, 1.0)
    columns = df.columns
    # ids = df.select(columns[0])
    # ids.save("ids_kmeans")
    vectors = df.select(columns[1:71])
    # maak random
    vectors = vectors.map(lambda x: Vectors.dense(x))
    data = vectors
    # vectors_rdd = vectors.rdd
    # vectors_rdd = vectors_rdd.map(lambda x: Vectors.dense(x))

    models = None


    wssses = []
    for n_clusters in range(480,500,20):
        # Build the model (cluster the data)
        models = KMeans.train(data, n_clusters, maxIterations=10, runs=10, initializationMode="random")

        # Evaluate clustering by computing Within Set Sum of Squared Errors
        def error(point):
            center = models.centers[models.predict(point)]
            # http://stackoverflow.com/questions/32977641/index-out-of-range-in-spark-mllib-k-means-with-tfidf-for-text-clutsering
            return sqrt(sum([x**2 for x in (point - center)]))
            # return sqrt(sum([x ** 2 for x in (point.toArray() - center)]))


        WSSSE = data.map(lambda point: error(point)).reduce(lambda x, y: x + y)
        logger.info("Within Set Sum of Squared Error = " + str(n_clusters) + "&" +  str(WSSSE) + "\\")
        wssses.append([n_clusters, WSSSE])

    # Save and load model
    models.save(sc, "w2v_model_kmeans")

    wssse_df = sqlContext.createDataFrame(wssses, ["n_clusters", "WSSSE"])
    wssse_df.save("wssse")

def predict():
    from pyspark.mllib.clustering import KMeans, KMeansModel
    from pyspark.mllib.linalg import Vectors
    model = KMeansModel.load(sc, "hdfs:///user/rmusters/w2v_model_kmeans")
    data = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").load("w2v_vector.csv")
    data = data.map(lambda x: [float(a) for a in x])
    df = data.toDF()
    columns = df.columns
    vectors = df.select(columns[1:71])
    vectors = vectors.map(lambda x: Vectors.dense(x))
    predicted = model.predict(vectors)
    result = predicted.map(lambda x: (x,)).toDF()
    result.save("clusters")
    # df = data.toDF()
    # columns = df.columns
    # vectors = df.select(columns[1:71])
    # vectors = vectors.map(lambda x: [float(a) for a in x])
    # vectors_rdd = vectors_rdd.map(lambda x: Vectors.dense(x))
    # vectors = sqlContext.createDataFrame(vectors, ["id", "vector"])

    # data = vectors.map(lambda (id, vectors): (id, vectors, model.predict(vectors)))
    # df = data.toDF(columns[1:71].append("cluster"))
    # df = df.select("cluster", "id")
    # df = df.sort(df.cluster.asc())
    # df.write.format("com.databricks.spark.csv").mode("overwrite").save("lambert_w2v_data_cluster.csv")
    # df.write.parquet("hdfs:///user/rmusters/lambert_w2v_data_cluster", mode="overwrite")
predict()
# train()