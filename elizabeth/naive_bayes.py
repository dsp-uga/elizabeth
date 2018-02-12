from pyspark.sql.functions import avg
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.feature import CountVectorizer, IDF

import elizabeth


def main(train_x, train_y, test_x, test_y=None, idf=False, base='gs', asm=False):
    # Load : DF[id, url, text, tokens, label?]
    # The DataFrames only have a labels column if labels are given.
    kind = 'asm' if asm else 'bytes'
    train = elizabeth.preprocess.load(train_x, train_y, base=base, kind=kind)
    test = elizabeth.preprocess.load(test_x, test_y, base=base, kind=kind)

    # TF : DF[id, url, text, tokens, label?, tf]
    tf = CountVectorizer(inputCol='tokens', outputCol='tf').fit(train)
    train, test = tf.transform(train), tf.transform(test)
    feature = 'tf'

    # IDF : DF[id, url, text, tokens, label?, tf, tfidf]
    if idf:
        idf = IDF(inputCol='tf', outputCol='tfidf').fit(train)
        train, test = idf.transform(train), idf.transform(test)
        feature = 'tfidf'

    # Naive Bayes : DF[id, url, text, tokens, label?, tf, tfidf, rawPrediction, probability, prediction]
    nb = NaiveBayes(featuresCol=feature, labelCol='label').fit(train)
    test = nb.transform(test)
    test = test.withColumn('prediction', test.prediction + 1)

    # If labels are given for the test set, print a score.
    if test_y:
        test = test.orderBy(test.id)
        test = test.withColumn('correct', (test.label == test.prediction).cast('double'))
        test = test.select(avg(test.correct))
        print(test.show())

    # If no labels are given for the test set, print predictions.
    else:
        test = test.orderBy(test.id).select(test.prediction)
        test = test.rdd.map(lambda row: int(row.prediction))
        test = test.toLocalIterator()
        print(*test, sep='\n')
