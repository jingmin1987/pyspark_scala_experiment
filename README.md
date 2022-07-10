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
### Spark setup
To be updated
### PySpark and Scala setup
To be updated