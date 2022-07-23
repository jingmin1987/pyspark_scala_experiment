
ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.13.8"

libraryDependencies += "org.scala-lang" % "scala-library" % "2.13.8"

libraryDependencies += "org.apache.spark" %% "spark-core" % "3.3.0"

lazy val root = (project in file("."))
  .settings(
    name := "scala-util"
  )
