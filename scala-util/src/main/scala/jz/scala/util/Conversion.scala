package jz.scala.util

import java.util
import java.util.{List => JList, Map => JMap, Set => JSet}
import scala.collection.JavaConverters._

object Conversion {
  def toSeq[A](pythonList: JList[A]): Seq[A] = pythonList.asScala
  def toMap[A, B](pythonMap: JMap[A, B]): Map[A, B] = pythonMap.asScala.toMap
  def toSet[A](pythonSet: JSet[A]): Set[A] = pythonSet.asScala.toSet

  def isSeq(mySeq: Any): Boolean = mySeq.isInstanceOf[Seq[_]]
  def isMap(myMap: Any): Boolean = myMap.isInstanceOf[Map[_, _]]
  def isSet(mySet: Any): Boolean = mySet.isInstanceOf[Set[_]]

  def testSeq():Unit = {
    val javaArrayList = new util.ArrayList[Int](Seq(1, 2).asJava)
    assert(isSeq(toSeq(javaArrayList)))

    val javaLinkedList = new util.LinkedList[Int](Seq(1, 2).asJava)
    assert(isSeq(toSeq(javaLinkedList)))
  }

  def testMap(): Unit = {
    val javaMap = new java.util.HashMap[String, Int] {
      put("first", 1)
      put("second", 2)
    }
    assert(isMap(toMap(javaMap)))
  }

  def testSet(): Unit = {
    val javaSet = new java.util.HashSet[Int]()
    javaSet.add(1)
    javaSet.add(2)
    javaSet.add(1)
    assert(isSet(toSet(javaSet)))
  }

  def main(args: Array[String]): Unit = {
    println(testSeq())
    println(testMap())
    println(testSet())
  }
}