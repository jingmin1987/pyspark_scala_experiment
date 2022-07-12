# Pyspark and Scala Experiment

## Idea
The simple idea is to create a machine learning platform that can have two backends: local Python and PySpark/Scala. More specifically, I want to find a way to expose Scala classes and functions to the frontend Jupyter Notebook directly. Something like this:

**Case 1** Using Python backend which is the default and it should utilize the local resources such as CPU and/or GPU
```python
from backend import BackendConfig # This is optional since it does not do anything
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier


data = pd.read_csv('my_file.csv', header=True)
gbt_model = GradientBoostingClassifier()
gbt_model.fit(data['X'], data['Y'])

# Of course, someone should still be able to use Spark in conjunction with Python
data2 = spark.read.parquet("/tenants/tenant1/dataset/pii/shared/my_data.parquet").toPandas()
gbt_model.fit(data2['X'], data2['Y'])

``` 
**Case 2** Using Scala backend which should utilize the cluster resources
```python
from backend import BackendConfig
from backend.scala.algo import XGBoostClassifier
from backend.scala.udf import MAP, VectorAssembler


BackendConfig.set_backend('scala')
BackendConfig.set_spark(spark=spark) # This should extract both sparkContext and active JVM
data2 = spark.read.parquet("/tenants/tenant1/dataset/pii/shared/my_data.parquet")
data_train = VectorAssembler(data=data_2, input_cols=data_2.columns, output_cols='features')
param = {
    'eta': 0.1,
    'missing': -999,
    'objective': 'multi:softprob',
    'num_class': 3
}
xgb_model = XGBoostClassifier(MAP(param)).fit(data_train) # A scala object

``` 

## Notes
### HDFS setup
I followed this [link](https://kontext.tech/article/445/install-hadoop-330-on-windows-10-using-wsl) to set up my local single-node HDFS
* Default DFS UI: http://localhost:9870/dfshealth.html#tab-overview
* Default YARN UI: http://localhost:8088/cluster
### Spark setup
I mainly followed this [link](https://kontext.tech/article/560/apache-spark-301-installation-on-linux-guide). However, a few links need to be updated accordingly
* The link to spark download returns 404 now. I've updated it to
```bash
wget https://dlcdn.apache.org/spark/spark-3.3.0/spark-3.3.0-bin-hadoop3.tgz
```
* All the following commands need to point to a slightly different location due to above change
* To run example. you should execute the below updated command instead
```bash
$SPARK_HOME/bin/run-example SparkPi 10
```
* Default SparkUI http://localhost:4040
* Default history server http://localhost:18080
### PySpark and Scala setup
Pyspark: just `pip install pyspark`
Scala: because I am using IntelliJ, so I only need to 
* Install JDK 1.8+
* Install Scala plugin for Intellij
* Config framework and SDK for the project
## Cheatsheet for HDFS and Spark
* Start DFS daemon `~/hadoop/hadoop-3.3.0$ sbin/start-dfs.sh`
* Check Java Process `jps`
* Start YARN daemon `~/hadoop/hadoop-3.3.0$ sbin/start-yarn.sh`
* Stop DFS daemon `~/hadoop/hadoop-3.3.0$ sbin/stop-dfs.sh`
* Stop YARN daemon `~/hadoop/hadoop-3.3.0$ sbin/stop-yarn.sh`
* Spark Shell `$ spark-shell`
* Jupyter Lab/Notebook without browser error message in WSL `$ jupyter lab --no-browser`
