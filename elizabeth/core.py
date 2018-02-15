import pyspark


def session(conf=None, **kwargs):
    '''Get or create the global `SparkSession`.

    Args:
        conf (SparkConf):
            The Spark configuration. Creates a new one if not given.

    Kwargs:
        Forwarded to `SparkConf.setAll` to initialize the session.

    Returns:
        SparkSession
    '''
    if not conf: conf = pyspark.SparkConf()
    conf.setAppName('elizabeth')
    conf.setAll(kwargs.items())
    sess = (pyspark.sql.SparkSession
        .builder
        .config(conf=conf)
        .getOrCreate())
    return sess
