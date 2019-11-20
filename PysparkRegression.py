from pyspark import SparkContext

sc = SparkContext("local", "First App")
# print(sc)
# print(sc.parallelize(range(1000)).count())
#
# import csv
# rdd = sc.textFile("vehicles.csv")
# rdd = rdd.mapPartitions(lambda x: csv.reader(x))

from pyspark.sql import SQLContext

sqlContext = SQLContext(sc)
data = sqlContext.read.option("inferSchema", "true").option("header", "true").csv("vehicles.csv")
data.show(4, False)
print('Notice The Difference Between how show and head works:  \n')
data.head(4)
print('Data type of the file is: ', type(data))  # pyspark.sql.dataframe
print('Number of rows in the file are:', data.count())
# WAYS TO CONVERT SQL DATAFRAME TO PANDAS DATAFRAME
# vehicles =  data.toPandas()
# print(vehicles.head(4))
print(data.dtypes)
data.printSchema()
print(data.columns)
data = data.drop('Make', 'Model', 'Vehicle Class')
data.show(4, False)

print(data.describe().show())

data.select('Transmission').show(5)
print('Unique values in Transmission are:', data.select('Transmission').distinct())
print('Number of Unique values in Transmission are:', data.select('Transmission').distinct().count())

print("We can also check two categorical columns as cross-tab")
data.crosstab('Drivetrain', 'Fuel Type').show()

data_copy = data  # Here I am storing a copy of the data before starting to clean the data
##################   MORE DATA CLEANING ###########################################################
# from pyspark.sql.functions import col, column

from pyspark.sql.functions import lower, col

data = data.withColumn("Year", col("Year").cast("int"))
# data.select('Transmission').distinct().show()
data.select('Transmission').distinct().count()


data = data.withColumn('Transmission_lower', lower(col('Transmission')))
data = data.withColumn('Drivetrain_lower', lower(col('Drivetrain')))
data = data.withColumn('FuelType_lower', lower(col('Fuel Type')))
data = data.drop('Transmission', 'Drivetrain', 'Fuel Type')
data.show(10)

from pyspark.sql.functions import udf
from pyspark.sql.types import StringType


def cleanTransmission(x):
    if 'auto' in x:
        return 'Automatic'
    else:
        return 'Manual'


def cleanFuel(x):
    if 'premium' in x or 'regular' in x or 'midgrade' in x or 'gasoline' in x:
        return 'gas'
    elif 'elect' in x:
        return 'electric'
    elif 'cng' in x:
        return 'cng'
    else:
        return 'diesel'


def cleanDrive(x):
    if 'rear' in x:
        return 'rearwheeldrive'
    elif 'front' in x:
        return 'frontwheeldrive'
    else:
        return 'allwheel'


funcTransmission = udf(cleanTransmission, StringType())
funcFuel = udf(cleanFuel, StringType())
funcDrive = udf(cleanDrive, StringType())
data = data.withColumn("Transmission", funcTransmission(data['Transmission_lower']))
data = data.withColumn("FuelType", funcFuel(data['FuelType_lower']))
data = data.withColumn("Drive", funcDrive(data['Drivetrain_lower']))
data = data.drop('Transmission_lower', 'Drivetrain_lower', 'FuelType_lower')
data.show(10)

# Year is taken as categorical variable. Next is cleaning the Year column as there are too many categories
data.select('Year').describe()
data.select('Year').distinct()
data.select('Year').distinct().count()
data.groupBy('Year').count().show()
data.groupBy('Year').count().sort('Year')


def cleanYear(x):
    if 1984 <= x <= 1990:
        return 'category1980-1990'
    elif 1991 <= x <= 1995:
        return 'category1990-1995'
    elif 1996 <= x <= 2000:
        return 'category1995-2000'
    else:
        return 'category2000-2003'


funcYear = udf(cleanYear, StringType())
data = data.withColumn("YearCategory", funcYear(data['Year']))
data = data.drop('Year')
data.show(10)

# Cleaning the cylinders columns : Deleting 2 and 3 cylinders cars
data = data.filter((data['Cylinders'] != '2.0') & (data['Cylinders'] != '3.0'))
data.show(4)

from pyspark.ml.feature import OneHotEncoder, StringIndexer
stringIndexer = StringIndexer(inputCol="Transmission", outputCol="TransmissionIndex")
model = stringIndexer.fit(data)
indexed = model.transform(data)
encoder = OneHotEncoder(inputCol="TransmissionIndex", outputCol="TransmissionVector")
encoded = encoder.transform(indexed)
encoded.show(4)
#(1,[0],[1.0]) means a vector of length 1 with 1.0 at position 1 and 0 elsewhere

from pyspark.ml import Pipeline
cols = ['Transmission', 'FuelType', 'Drive', 'YearCategory']
print(cols)
indexers = [StringIndexer(inputCol=column, outputCol=column+"Index").fit(data) for column in cols]
pipeline = Pipeline(stages=indexers)
dataIndexed = pipeline.fit(data).transform(data)
cols = [column+"Index" for column in cols]
encoders = [OneHotEncoder(inputCol=column, outputCol=column+"Vector") for column in cols]
pipeline = Pipeline(stages=encoders)
dataVectors = pipeline.fit(dataIndexed).transform(dataIndexed)
dataVectors.show(4)

dataFinal = dataVectors.drop('Transmission', 'FuelType', 'Drive', 'YearCategory', 'FuelTypeIndex', 'DriveIndex', 'YearCategoryIndex', 'TransmissionIndex')
print('Data Final is :')
dataFinal.show(10)
from pyspark.ml.feature import VectorAssembler
va = VectorAssembler(inputCols=['Engine Displacement','Cylinders', 'Fuel Barrels/Year', 'City MPG', 'Highway MPG'\
'Combined MPG', 'Fuel Cost/Year', 'TransmissionIndexVector',\
'FuelTypeIndexVector', 'DriveIndexVector', 'YearCategoryIndexVector'], outputCol='features')

from pyspark.ml.regression import RandomForestRegressor
rf = RandomForestRegressor(featuresCol='features', labelCol='CO2 Emission Grams/Mile', numTrees=10)

from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator











sc.stop()