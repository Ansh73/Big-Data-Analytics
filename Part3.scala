import org.apache.spark.mllib.fpm.FPGrowth
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}
object Part3 {
  def main(args: Array[String]): Unit = {
    val sc = new SparkContext(new SparkConf().setAppName("Part3")/*.setMaster("local[2]")*/)
    val spark = SparkSession.builder().getOrCreate()
    val data = sc.textFile(args(0))

    val headerData = data.first
    val cleanData = data.filter(line => line != headerData)
    val orderProductID = cleanData.map(x => ((x.split(",")) (0), (x.split(",")) (1)))

    val orderProductDetails = orderProductID.groupByKey.mapValues(_.toArray)
    val products = orderProductDetails.map(x => x._2)

    val fpg = new FPGrowth().setMinSupport(0.005).setNumPartitions(10)
    val model = fpg.run(products)

    var result = ""
    result+="*Frequent Itemsets*\n"
    model.freqItemsets.sortBy(-_.freq).take(10).foreach { itemset =>
      result+=itemset.items.mkString("[", ",", "]")
      result += ", " + itemset.freq+"\n"
    }
    result+="\n\n---------------------------------------------------------------------\n"
    result+="Association Rules: \n"
    val minConfidence = 0.3
    model.generateAssociationRules(minConfidence).sortBy(-_.confidence).take(10).foreach { rule =>
      result += rule.antecedent.mkString("[", ",", "]") + " => "
      result += rule.consequent.mkString("[", ",", "]") + ", " + rule.confidence+"\n"
    }
    sc.parallelize(List(result)).saveAsTextFile(args(1))
  }
}