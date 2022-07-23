package jz.scala.util

import java.util
import java.util.{List => JList}
import java.util.{HashMap => JMap}
import scala.jdk.CollectionConverters._

object Conversion {
  def toSeq[A](pythonList: JList[A]): Seq[A] = pythonList.asScala.toSeq
  def toMap[A, B](pythonMap: JMap[A, B]): Map[A, B] = pythonMap.asScala.toMap

  def testSeq():Seq[Int] = {
    val javaArrayList = new util.ArrayList[Int](Seq(1, 2).asJava)
    toSeq(javaArrayList)

    val javaLinkedList = new util.LinkedList[Int](Seq(1, 2).asJava)
    toSeq(javaLinkedList)
  }

  def testMap(): Map[String, Int] = {
    val javaMap = new JMap[String, Int] {
      put("first", 1)
      put("second", 2)
    }
    toMap(javaMap)
  }

  def main(args: Array[String]): Unit = {}
}