# Pyspark and Spark Experiment

## Why did I create this project?
I noticed that some team's work flow is segmented as below, mainly due to constraints in different environments
* Preprocessing happens in Python and PySpark.
* Model training is written in Scala and run on Spark using spark-submit.
* Mainly rely on grid search for hyperparameter tuning due to inconvenience of visualizing the parameter space while 
  running the model using spark-submit
* Performance reporting and other postprocessing are in Python and Pyspark.
 
Therefore, the idea of streamlining model training onto one platform (e.g. Jupyter) and make it more 
enjoyable/interactive is the core motivation to have this toy project. Ideally, user should have almost seamless 
experience when switching between Python and Spark backend, with almost the same APIs when building the model. 
Gradually more functionality could be implemented on demand.

The realization of the idea is to leverage `py4j` and implement the wrapper of xgboost-spark on the Python side. 
This idea is different from what was implemented in `xgboost-pyspark` which utilizes pyspark.ml. Anyway, I 
think it is a good fun project to implement it in a different way, and learn how to program across languages and 
environments  .

**Example 1** Using Python backend which is the default, and it should utilize the local resources such as CPU 
and/or GPU
```python
import pandas as pd
import optuna
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from estimator.xgbclassifier import XGBClassifier

# Global vars
SEED = 123
FIXED_PARAM = {
  'eval_metric': 'auc',
}

# Create fake data
x, y = make_classification(n_samples=10_000)
x = pd.DataFrame(data=x, columns=[f'feature_{i}' for i in range(x.shape[1])])
x_train, x_valid, y_train, y_valid = train_test_split(x, y, train_size=0.7, random_state=SEED)

# Create XGBoost classifier python model
def objective(trial):
  global FIXED_PARAM, x_train, x_valid, y_train, y_valid
  param = {
    'learning_rate': trial.suggest_float('learning_rate', 1e-8, 1.0, log=True)
  }

  python_clf = XGBClassifier.make(**FIXED_PARAM, **param)
  return python_clf.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False).best_score

# And optimize hyperparameters in Python
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)
``` 
**Example 2** Using Spark backend which should utilize the cluster resources
```python
import pandas as pd
import optuna
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from pyspark.sql import SparkSession # New

from estimator.xgbclassifier import XGBClassifier

# Global vars
SEED = 123
FIXED_PARAM = {
  'eval_metric': 'auc',
}

# Spark if there's none
spark = SparkSession.builder.appName('my_test').config('spark.jars', '../jar/scala-util.jar,../jar/xgboost4j_2.12-1.6.1.jar,../jar/xgboost4j-spark_2.12-1.6.1.jar').getOrCreate()

# Create fake data
x, y = make_classification(n_samples=10_000)
x = pd.DataFrame(data=x, columns=[f'feature_{i}' for i in range(x.shape[1])])
xy = x.assign(label=y)
xy_train, xy_valid = map(spark.createDataFrame, train_test_split(xy, train_size=0.7, random_state=SEED)) # to Spark
FIXED_PARAM['eval_sets'] = {'eval1': xy_valid} # Different way of adding 'eval_sets'
FIXED_PARAM['verbose'] = False

# Create XGBoost classifier spark model

def objective(trial):
  global FIXED_PARAM, x_train, x_valid, y_train, y_valid
  param = {
    'learning_rate': trial.suggest_float('learning_rate', 1e-8, 1.0, log=True)
  }

  spark_clf = XGBClassifier \
    .make(backend='spark', spark=spark, **FIXED_PARAM, **param)

  vectorized_df = spark_clf.transform(xy_train, xy_train.columns[:-1])

  return float(spark_clf.fit(vectorized_df).booster.getAttr('best_score'))

# And optimize hyperparameters in Python
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

``` 
## Testing System
* Spark 3.3.0 - standalone mode
* Hadoop 3.3.0
* Scala 2.12
* Java 1.8
* Python 3.8
* Packages - please refer to environment/requirements.txt
## Notes in Regard to Performance
* Using spark, `num_workers>1` has significant overhead. For my toy dataset, it is not worth going over 1
* Using spark with `num_workers=1`, the performance seems about 50% worse than using Python with `n_jobs=8` with the 
  toy dataset. Supposedly the overhead will become worthwhile once data becomes large enough
* Hasn't yet tested in a distributed system, but I was forewarned that one should set `--executor-cores` to be 1 
* For standalone model, `driver memory` is what matters and may impact how much shuffling there could be
* Other parameters such as `number of partitions` could be further optimized. But I don't really think it's 
  necessary for this project
* As far as I know, it is preferred to call methods of Spark DataFrame than methods of PythonRDD
## Other Notes
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
* To compile as JAR in Intellij, one can add an JAR artifact under File -> Project Structure and then build the 
  artifact
* To test the JAR file, one can run `java -cp out/.../scala_test.jar test.Hello`
## Cheatsheet for HDFS and Spark
* For some reason I have to constantly `sudo service ssh restart`
* Start DFS daemon `~/hadoop/hadoop-3.3.0$ sbin/start-dfs.sh`
* Check Java Process `jps`
* Start YARN daemon `~/hadoop/hadoop-3.3.0$ sbin/start-yarn.sh`
* Stop DFS daemon `~/hadoop/hadoop-3.3.0$ sbin/stop-dfs.sh`
* Stop YARN daemon `~/hadoop/hadoop-3.3.0$ sbin/stop-yarn.sh`
* Spark Shell `$ spark-shell`
* Jupyter Lab/Notebook without browser error message in WSL `$ jupyter lab --no-browser`
## Useful links
https://www.waitingforcode.com/pyspark/pyspark-jvm-introduction-1/read
