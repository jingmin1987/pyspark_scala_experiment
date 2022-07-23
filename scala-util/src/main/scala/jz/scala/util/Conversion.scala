package jz.scala.util

import java.util
import java.util.{List => JList}
import java.util.{Map => JMap}
import scala.collection.JavaConverters._

object Conversion {
  def toSeq[A](pythonList: JList[A]): Seq[A] = pythonList.asScala
  def toMap[A, B](pythonMap: JMap[A, B]): Map[A, B] = pythonMap.asScala.toMap

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

  def main(args: Array[String]): Unit = {
    println(testSeq())
    println(testMap())
  }
}