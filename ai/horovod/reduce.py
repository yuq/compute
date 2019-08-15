import itertools
import tensorflow as tf
import horovod.tensorflow as hvd

hvd.init()
size = hvd.size()
local_rank = hvd.rank()

def random_uniform(*args, **kwargs):
    if hasattr(tf, 'random') and hasattr(tf.random, 'set_seed'):
        tf.random.set_seed(1234)
        return tf.random.uniform(*args, **kwargs)
    else:
        tf.set_random_seed(1234)
        return tf.random_uniform(*args, **kwargs)

dtypes = [tf.float32]
dims = [1]

for dtype, dim in itertools.product(dtypes, dims):
    with tf.device("/cpu:0"):
        tensor = random_uniform(
            [17] * dim, -100, 100, dtype=dtype)
        summed = hvd.allreduce(tensor, average=False)
    multiplied = tensor * size
    max_difference = tf.reduce_max(tf.abs(summed - multiplied))

    # Threshold for floating point equality depends on number of
    # ranks, since we're comparing against precise multiplication.
    if size <= 3 or dtype in [tf.int32, tf.int64]:
        threshold = 0
    elif size < 10:
        threshold = 1e-4
    elif size < 15:
        threshold = 5e-4
    else:
        break

    with tf.Session() as sess:
        diff = sess.run(max_difference)

    if diff <= threshold:
        print("rank %d pass" % local_rank)
    else:
        print("rank %d fail" % local_rank)
