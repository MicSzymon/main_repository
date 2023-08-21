"""
funnel type of loss function proposal (reverse than Huber loss)
could be useful to automatic learning of most common examples and marginalize impact of single events.
I know that similar effect could be reached by using loss wages but there are situations that loss wages could not be used.
Correct numbers are necessary for function continuity at delta and -delta.
"""

def funnel_loss(y_true, y_pred, delta=0.5):
    error = y_true - y_pred
    abs_error = tf.abs(error)
    quad = tf.where(abs_error < delta, 8*tf.square(abs_error), 2+delta * (0.5*abs_error - 0.5 * delta))
    return tf.reduce_mean(quad)

 model.compile(loss=funnel_loss,
                optimizer=tf.optimizers.Adamax(),
                metrics=[tf.metrics.MeanAbsoluteError()])