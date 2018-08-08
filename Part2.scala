import org.apache.spark.sql.functions._
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.mllib.clustering.LDA
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import scala.collection.mutable

object Part2 {

  def main(args: Array[String]): Unit = {

    var result = ""

    val sc = new SparkContext(new SparkConf().setAppName("TopicModeling")/*.setMaster("local[2]")*/)
    val spark = SparkSession.builder().getOrCreate()

    val sqlContext = SparkSession.builder().getOrCreate()
    //val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    import sqlContext.implicits._

    val airlines = spark.read.option("header", "true").option("inferSchema", "true").csv(args(0))

    //Created DataFrame where airline text field does not have NULL value
    val sentiments = airlines.filter($"text".isNotNull).toDF()
    //Selected unique values from airline_sentiment column
    sentiments.select("airline_sentiment").distinct.show()
    //Converted sentiment values from String to Numeric
    val sentiment_Numeric = sentiments.withColumn("airline_sentiment", when(col("airline_sentiment") === "positive", 5.0).otherwise(col("airline_sentiment"))).withColumn("airline_sentiment", when(col("airline_sentiment") === "negative", 1.0).otherwise(col("airline_sentiment"))).withColumn("airline_sentiment", when(col("airline_sentiment") === "neutral", 2.5).otherwise(col("airline_sentiment")))
    //Calculated Average of each airlines' sentiment values
    val avgRating = sentiment_Numeric.groupBy("airline").agg(avg("airline_sentiment"))

    //Ordered the Average values in descending way
    val descOrderedAirlines = avgRating.orderBy($"avg(airline_sentiment)".desc).toDF()
    //First one is the Best Airline
    val Best = descOrderedAirlines.first.getString(0)
    val BestRating = descOrderedAirlines.first.getDouble(1)

    //Ordered the Average values in ascending way
    val orderedAirlines = avgRating.orderBy($"avg(airline_sentiment)").toDF()
    //First one is the Worst Airline
    val Worst = orderedAirlines.first.getString(0)
    val WorstRating = orderedAirlines.first.getDouble(1)

    result += "The Best Airline: " + Best + " having Ratings " + BestRating + "\n"
    result += "The Worst Airline: " + Worst + " having Ratings " + WorstRating + "\n"
    result += "---------------------------------------------------------------------"

    val stopWords = StopWordsRemover.loadDefaultStopWords("english").toSet

    //Get the airline data
    val flightData = sc.textFile(args(0))
    //Get the Header
    val dataHeader = flightData.first
    //Filter all data with ',' separated format except header
    val feedback = flightData.filter(line => line != dataHeader && line.split(",").length > 10)
    //Get Pair RDD from the filtered data
    val pairRDD = feedback.map(s => ((s.split(",")) (5), (s.split(",")) (10)))
    //Select Base and Worst Airlies data
    //val BestWorstAirlines = pairRDD.filter(b => (b._1 == Best || b._1 == Worst ))
    val bestAirline = pairRDD.filter(bestAir => bestAir._1 == Best)
    val worstAirline = pairRDD.filter(worstAir => worstAir._1 == Worst)

    //Created text Corpus
    val corpus_best = bestAirline.map(w => w._2)
    //Tokenized the lines
    val tokenized_best: RDD[Seq[String]] =
      corpus_best.map(_.toLowerCase.split("\\s")).map(_.filter(_.length > 3).filter(token => !stopWords.contains(token)).filter(_.forall(java.lang.Character.isLetter)))
    //Get a Term Count
    val termCounts_best: Array[(String, Long)] =
      tokenized_best.flatMap(_.map(_ -> 1L)).reduceByKey(_ + _).collect().sortBy(-_._2)
    //In the above RDD you have terms sorted by their frequency
    //Get rid of the top 25 most frequent words -> likely stopwords
    val mostFrquentWords = 25
    val vocabArray: Array[String] =
      termCounts_best.takeRight(termCounts_best.size - mostFrquentWords).map(_._1)

    //Get the useful vocabulary:
    val vocab: Map[String, Int] = vocabArray.zipWithIndex.toMap

    //Created a term frequency vector for each document:
    val documents: RDD[(Long, Vector)] =
      tokenized_best.zipWithIndex.map { case (tokens, id) =>
        val counts = new mutable.HashMap[Int, Double]()
        tokens.foreach { term =>
          if (vocab.contains(term)) {
            val idx = vocab(term)
            counts(idx) = counts.getOrElse(idx, 0.0) + 1.0
          }
        }
        (id, Vectors.sparse(vocab.size, counts.toSeq))
      }

    //Created LDA model
    val K = 6
    val lda = new LDA().setK(K).setMaxIterations(30)
    //Run the model on the documents and examine results:
    val ldaModel = lda.run(documents)

    //Print topics, showing top-weighted 10 terms for each topic
    val topicIndices = ldaModel.describeTopics(maxTermsPerTopic = 10)
    topicIndices.foreach { case (terms, termWeights) =>
      result += "Best airlines Topic and its weight:\n"
      terms.zip(termWeights).foreach { case (term, weight) =>
        result += {
          vocabArray(term.toInt)
        }
        result += " | " + weight + ",\n"
      }
      result += "-----------------------------------------------------------------------\n"
    }
    //Best Airline Done

    //For the worst airlines

    //Created text Corpus
    val corpus_worst = worstAirline.map(w => w._2)
    //Tokenized the lines
    val tokenized_worst: RDD[Seq[String]] =
      corpus_worst.map(_.toLowerCase.split("\\s")).map(_.filter(_.length > 3).filter(token => !stopWords.contains(token)).filter(_.forall(java.lang.Character.isLetter)))
    //Get a Term Count
    val termCounts_worst: Array[(String, Long)] =
      tokenized_worst.flatMap(_.map(_ -> 1L)).reduceByKey(_ + _).collect().sortBy(-_._2)
    //In the above RDD you have terms sorted by their frequency
    //Get rid of the top 25 most frequent words -> likely stopwords
    val mostFrequentWords_1 = 25
    val vocabArray_worst: Array[String] =
      termCounts_worst.takeRight(termCounts_worst.size - mostFrequentWords_1).map(_._1)

    //Get the useful vocabulary:
    val vocab_worst: Map[String, Int] = vocabArray_worst.zipWithIndex.toMap

    //Created a term frequency vector for each document:
    val documents_worst: RDD[(Long, Vector)] =
      tokenized_worst.zipWithIndex.map { case (tokens, id) =>
        val counts = new mutable.HashMap[Int, Double]()
        tokens.foreach { term =>
          if (vocab_worst.contains(term)) {
            val idx = vocab_worst(term)
            counts(idx) = counts.getOrElse(idx, 0.0) + 1.0
          }
        }
        (id, Vectors.sparse(vocab_worst.size, counts.toSeq))
      }

    //Created LDA model
    val K_worst = 6
    val lda_worst = new LDA().setK(K_worst).setMaxIterations(30)
    //Run the model on the documents and examine results:
    val ldaModel_worst = lda_worst.run(documents_worst)

    //Print topics, showing top-weighted 10 terms for each topic
    val topicIndices_worst = ldaModel_worst.describeTopics(maxTermsPerTopic = 10)
    topicIndices_worst.foreach { case (terms, termWeights) =>
      result += "Worst airlines Topic and its weight:\n"
      terms.zip(termWeights).foreach { case (term, weight) =>
        result += {
          vocabArray_worst(term.toInt)
        }
        result += " | " + weight + ",\n"
      }
      result += "-----------------------------------------------------------------------\n"
    }
  //Worst airline done
    sc.parallelize(List(result)).saveAsTextFile(args(1))
  }
}