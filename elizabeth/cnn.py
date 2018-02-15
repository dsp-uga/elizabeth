from pyspark.ml import Estimator, Transformer
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.feature import CountVectorizer, Word2Vec
from pyspark.ml.param import Param
from pyspark.sql.functions import avg, max as max_, size, udf
from pyspark.sql.types import ArrayType, FloatType

from bigdl.models.ml_pipeline.dl_classifier import DLClassifier
from bigdl.nn.criterion import CategoricalCrossEntropy
from bigdl.nn.layer import Sequential, Linear, Padding, TemporalConvolution as Conv1D, ReLU
from bigdl.util.common import create_spark_conf, init_engine

import elizabeth


class HexToFloat(Transformer):
    '''Transforms an array of hex byte strings to floats in the range [0,1].
    '''
    def __init__(self, inputCol='features', outputCol='transform'):
        super().__init__()
        self.inputCol = Param(self, 'inputCol', 'input column name.')
        self.outputCol = Param(self, 'outputCol', 'output column name.')
        self.setParams(inputCol=inputCol, outputCol=outputCol)

    def setParams(self, **kwargs):
        return self._set(**kwargs)

    def _transform(self, df):
        in_col = self.getOrDefault('inputCol')
        out_col = self.getOrDefault('outputCol')
        transform = lambda bytes: [int(b, 16) / 255 for b in bytes]
        transform = udf(transform, ArrayType(FloatType()))
        return df.withColumn(out_col, transform(df[in_col]))


def main(train_x, train_y, test_x, test_y=None, word2vec=False, base='gs', asm=False):
    # BigDL requires a special SparkConf.
    conf = create_spark_conf()
    spark = elizabeth.session(conf)
    init_engine()

    # Load : DF[id, url, text, features, label?]
    # The DataFrames only have a labels column if labels are given.
    kind = 'asm' if asm else 'bytes'
    train = elizabeth.load(train_x, train_y, base=base, kind=kind)
    test = elizabeth.load(test_x, test_y, base=base, kind=kind)

    # Get the size of the longest document.
    n = train.select(max_(size(train.features))).head()[0]

    # Get the number of classes.
    n_classes = 9  # TODO: don't hard code

    # Train the preprocessor and transform the data.
    prep = elizabeth.Preprocessor()
    if word2vec: prep.add(Word2Vec())
    else: prep.add(HexToFloat())
    train = prep.fit(train)
    test = prep.transform(test)

    # Define the neural net.
    model = Sequential()
    model.add(Padding(1, n, 2))
    model.add(Conv1D(n//1, n//2, 5, 2)).add(ReLU())
    model.add(Conv1D(n//2, n//4, 5, 2)).add(ReLU())
    model.add(Conv1D(n//4, n//8, 5, 2)).add(ReLU())
    model.add(Linear(n//8, 256)).add(ReLU())
    model.add(Linear(256, n_classes))
    criterion = CategoricalCrossEntropy()
    cnn = DLClassifier(model, criterion, [n])

    # Train the model.
    cnn.fit(train)
    test = cnn.transform(test)
    test = test.withColumn('prediction', test.prediction + 1)

    # If labels are given for the test set, print a score.
    if test_y:
        score = test.orderBy(test.id)
        score = score.withColumn('correct', (score.label == score.prediction).cast('double'))
        score = score.select(avg(score.correct))
        score.show()

    # If no labels are given for the test set, print predictions.
    else:
        # test = test.orderBy(test.id).select(test.prediction)
        # test = test.rdd.map(lambda row: int(row.prediction))
        # test = test.toLocalIterator()
        # print(*test, sep='\n')
        test.show()
