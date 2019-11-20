from pyspark import SparkContext, SparkConf
import collections

conf = SparkConf().setMaster('local').setAppName('RatingsHistogram')
sc = SparkContext(conf=conf)

# lines = sc.textFile('file:///Users/himanshuaggarwal/PycharmProjects/pyspark_projects/sparkCourse/ml-100k/u.data')
lines = sc.textFile('file:///sparkCourse/ml-100k/u.data')
ratings = lines.map(lambda x: x.split()[2])
results = ratings.countByValue()
sortedResults = collections.OrderedDict(sorted(results.items()))
for key, value in sortedResults.items():
    print("%s %i" % (key, value))


