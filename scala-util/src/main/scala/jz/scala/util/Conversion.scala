package jz.scala.util

import java.util
import java.util.{List => JList, Map => JMap, Set => JSet}
import scala.collection.JavaConverters._

object Conversion {
  def toSeq[A](pythonList: JList[A]): Seq[A] = pythonList.asScala
  def toMap[A, B](pythonMap: JMap[A, B]): Map[A, B] = pythonMap.asScala.toMap
  def toSet[A](pythonSet: JSet[A]): Set[A] = pythonSet.asScala.toSet

  def testSeq():Seq[Int] = {
    val javaArrayList = new util.ArrayList[Int](Seq(1, 2).asJava)
    toSeq(javaArrayList)

    val javaLinkedList = new util.LinkedList[Int](Seq(1, 2).asJava)
    toSeq(javaLinkedList)
  }

  def testMap(): Map[String, Int] = {
    val javaMap = new java.util.HashMap[String, Int] {
      put("first", 1)
      put("second", 2)
    }
    toMap(javaMap)
  }

  def testSet(): Set[Int] = {
    val javaSet = new java.util.HashSet[Int]()
    javaSet.add(1)
    javaSet.add(2)
    javaSet.add(1)
    toSet(javaSet)
  }

  def main(args: Array[String]): Unit = {
    println(testSeq())
    println(testMap())
    println(testSet())
  }
}