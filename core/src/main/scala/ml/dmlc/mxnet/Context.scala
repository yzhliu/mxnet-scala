package ml.dmlc.mxnet

/*
  Constructing a context.

  Parameters
  ----------
  device_type : {'cpu', 'gpu'} or Context.
      String representing the device type

  device_id : int (default=0)
      The device id of the device, needed for GPU

  Note
  ----
  Context can also be used a way to change default context.

  Examples
  --------
  >>> # array on cpu
  >>> cpu_array = mx.md.ones((2, 3))
  >>> # switch default context to GPU(2)
  >>> with mx.Context(mx.gpu(2)):
  >>>     gpu_array = mx.md.ones((2, 3))
  >>> gpu_array.context
  gpu(2)
*/
class Context(deviceTypeName: String, val deviceId: Int = 0) {
  val devtype2str = Map(1 -> "cpu", 2 -> "gpu", 3 -> "cpu_pinned")
  val devstr2type = Map("cpu" -> 1, "gpu" -> 2, "cpu_pinned" -> 3)

  val deviceTypeid: Int = devstr2type(deviceTypeName)

  def this(context: Context) = {
    this(context.deviceType, context.deviceId)
  }

  /*
    Return device type of current context.

    Returns
    -------
    device_type : str
  */
  def deviceType: String = devtype2str(deviceTypeid)
}
