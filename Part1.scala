import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.{Pipeline}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.{IDF, Tokenizer}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.classification.DecisionTreeClassifier

object Part1 {
  def main(args: Array[String]): Unit ={
    if (args.length == 0) {
      println("I need two parameters ")
    }

    val sc = new SparkContext(new SparkConf().setAppName("Part1")/*.setMaster("local[2]")*/)
    val spark = SparkSession.builder().getOrCreate()
    val sqlContext = SparkSession.builder().getOrCreate()
    var output = ""
    val originalDF = spark.read.option("header","true").option("inferSchema","true").csv(args(0))
    val updatedDF = originalDF.drop("tweet_id").drop("negativereason").drop("airline_sentiment_gold").drop("name").drop("negativereason_gold").drop("tweet_coord").drop("tweet_created").drop("tweet_location").drop("user_timezone")

    val wordTokenizer = new Tokenizer().setInputCol("text").setOutputCol("words")
    val stopWordsRemover = new StopWordsRemover().setInputCol("words").setOutputCol("importantWords")
    val termHasher = new org.apache.spark.ml.feature.HashingTF().setInputCol("importantWords").setOutputCol("rawFeatures").setNumFeatures(1000)
    val labelConverter = new StringIndexer().setInputCol("airline_sentiment").setOutputCol("airline_sentiment_index")

    val pipeline = new Pipeline().setStages(Array(wordTokenizer, stopWordsRemover, termHasher, labelConverter))

    import sqlContext.implicits._
    val filteredDF = updatedDF.filter($"text".isNotNull).toDF()
    val tokenizedDF = wordTokenizer.transform(filteredDF)
    val goodWordsDF = stopWordsRemover.transform(tokenizedDF)
    val hashedDF = termHasher.transform(goodWordsDF)
    val indexedDF= labelConverter.fit(hashedDF).transform(hashedDF)

    val cleanDF = indexedDF.na.fill(0, Seq("negativereason_confidence"))
    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("rawValues")
    val idfModel = idf.fit(cleanDF)
    val transformedCleanDF = idfModel.transform(cleanDF)
    val myDF = transformedCleanDF.drop("text").drop("words").drop("importantWords").drop("rawFeatures")
    val tempDF = myDF.withColumnRenamed( "airline_sentiment_index", "label").na.fill("0")
    val finalData = tempDF.select(tempDF("airline_sentiment_confidence").cast(IntegerType).as("airline_sentiment_confidence"), tempDF("negativereason_confidence").cast(IntegerType).as("negativereason_confidence"), tempDF("retweet_count"), tempDF("label").cast(IntegerType).as("label"), tempDF("rawValues"))

    val assembler = new VectorAssembler()
    assembler.setInputCols(Array("airline_sentiment_confidence", "negativereason_confidence", "retweet_count", "rawValues"))
    assembler.setOutputCol("features")

    val dt = new DecisionTreeClassifier().setLabelCol("label").setFeaturesCol("features").setMaxDepth(5)
    val pipeline_dt = new Pipeline().setStages(Array(assembler,dt))
    val paramGrid_dt = new ParamGridBuilder().addGrid(dt.maxDepth, Array(10,15,20)).build()
    val evaluator_dt = new MulticlassClassificationEvaluator()
    evaluator_dt.setLabelCol("label")
    evaluator_dt.setMetricName("accuracy")
    val cv_dt = new CrossValidator().setEstimator(pipeline_dt).setEstimatorParamMaps(paramGrid_dt).setEvaluator(evaluator_dt).setNumFolds(5)

    val Array(train, test) =finalData.randomSplit(Array(0.9,0.1))
    val model_dt = cv_dt.fit(train)
    val result_dt = model_dt.transform(test)
    val accuracy_dt = evaluator_dt.evaluate(result_dt)

    output+="Accuracy of Decision Tree  model is :"+accuracy_dt+"\n"
    output+="\n\n--------------------------------------------------------------------\n"
    output+="Using Logistic Regression\n"


    val logisticRegression = new LogisticRegression().setMaxIter(8).setRegParam(0.3).setElasticNetParam(0.8).setLabelCol("label").setFamily("multinomial")
    val pipeline_lr = new Pipeline().setStages(Array(assembler,logisticRegression))
    val paramGrid_lr = new ParamGridBuilder().addGrid(logisticRegression.threshold, Array(0.3,0.4,0.5)).addGrid(logisticRegression.regParam, Array(0.01,0.001, 0.0001)).addGrid(logisticRegression.maxIter, Array(10,20,30)).build()
    val evaluator_lr = new MulticlassClassificationEvaluator()
    evaluator_lr.setLabelCol("label")
    evaluator_lr.setMetricName("accuracy")
    val cv_lr = new CrossValidator().setEstimator(pipeline_lr).setEstimatorParamMaps(paramGrid_lr).setEvaluator(evaluator_lr).setNumFolds(5)

    val Array(train_lr, test_lr) =finalData.randomSplit(Array(0.9,0.1))
    val model = cv_lr.fit(train_lr)
    val result_lr = model.transform(test_lr)
    val accuracy_lr = evaluator_lr.evaluate(result_lr)
    output+="Accuracy of the model is :"+accuracy_lr+"\n"
    sc.parallelize(List(output)).saveAsTextFile(args(1))
  }
}