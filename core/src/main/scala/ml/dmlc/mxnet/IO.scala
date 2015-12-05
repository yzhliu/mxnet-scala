package ml.dmlc.mxnet

/*
object IO {
  DataBatch = namedtuple('DataBatch', ['data', 'label', 'pad', 'index'])

  // Convert data into canonical form.
  def initData(data: , allowEmpty: Boolean, defaultName: String) = {
    require(
    assert (data is not None) or allow_empty
    if data is None:
      data = []

    if isinstance(data, (np.ndarray, NDArray)):
      data = [data]
    if isinstance(data, list):
      if not allow_empty:
        assert(len(data) > 0)
      if len(data) == 1:
        data = OrderedDict([(default_name, data[0])])
      else:
        data = OrderedDict([('_%d_%s' % (i, default_name), d) for i, d in enumerate(data)])
    if not isinstance(data, dict):
        raise TypeError("Input must be NDArray, numpy.ndarray, a list of them or dict with them as values")
    for k, v in data.items():
      if isinstance(v, NDArray):
        data[k] = v.asnumpy()
    for k, v in data.items():
      if not isinstance(v, np.ndarray):
        raise TypeError(("Invalid type '%s' for %s, "  % (type(v), k)) + "should be NDArray or numpy.ndarray")
  }
}

trait DataIter {
  """DataIter object in mxnet. """

  def reset(): Unit

  /**
    Get next data batch from iterator
    Returns
    -------
    data : NDArray
      The data of next batch.
    label : NDArray
      The label of next batch.
   *
   */
  def next(): (NDArray, NDArray)

  /**
    Iterate to next batch.
    Returns
    -------
    has_next : boolean
        Whether the move is successful.
  */
  def iterNext(): Boolean

  /**
    Get data of current batch.
    Parameters
    ----------
    index : int
        The index of data source to retrieve.
    Returns
    -------
    data : NDArray
        The data of current batch.
  */
  def getdata(index: Int = 0): NDArray

  /**
    Get label of current batch.
    Returns
    -------
    label : NDArray
        The label of current batch.
  */
  def getlabel(): NDArray = getdata(-1)

  /**
    Retures
    -------
    index : numpy.array
        The index of current batch
  */
  def getindex(): Int

  /**
  Get the number of padding examples in current batch.
  Returns
  -------
  pad : int
      Number of padding examples in current batch
  */
  def getpad(): Int
}

/**
  NDArrayIter object in mxnet. Taking NDArray or numpy array to get dataiter.
  Parameters
  ----------
  data_list or data, label: a list of, or two separate NDArray or numpy.ndarray
    list of NDArray for data. The last one is treated as label.
  batch_size: int
    Batch Size
  shuffle: bool
    Whether to shuffle the data
  data_pad_value: float, optional
    Padding value for data
  label_pad_value: float, optionl
    Padding value for label
  last_batch_handle: 'pad', 'discard' or 'roll_over'
    How to handle the last batch
  Note
  ----
  This iterator will pad, discard or roll over the last batch if
  the size of data does not match batch_size. Roll over is intended
  for training and can cause problems if used for prediction.
*/
class NDArrayIter extends DataIter {

  def __init__(self, data, label= None, batch_size=1, shuffle= False, last_batch_handle='pad'):
  super(NDArrayIter, self).__init__()

  self.data = _init_data(data, allow_empty= False, default_name= 'data ')
  self.label = _init_data(label, allow_empty=True, default_name= 'softmax_label')

  // shuffle data
  if shuffle:
    idx = np.arange(self.data[0][1].shape[0])
    np.random.shuffle(idx)
    self.data = [(k, v[idx]) for k, v in self.data]
    self.label = [(k,v[idx]) for k, v in self.label]

  self.data_list = [x[1] for x in self.data] + [x[1] for x in self.label]
  self.num_source = len(self.data_list)

  // batching
  if last_batch_handle == 'discard ':
    new_n = self.data_list[0].shape[0] -self.data_list[0].shape[0] % batch_size
  for k, _ in self.data:
    self.data[k] = self.data[k][: new_n]
  for k, _ in self.label:
    self.label[k] = self.label[k][: new_n]
  self.num_data = self.data_list[0].shape[0]
  assert self.num_data >= batch_size, \
  "batch_size need to be smaller than data size when not padding."
  self.cursor = -batch_size
  self.batch_size = batch_size
  self.last_batch_handle = last_batch_handle

  @property
  def provide_data(self):
    """The name and shape of data provided by this iterator"""
    return [(k, tuple([self.batch_size] + list(v.shape[1: ] ) )) for k, v in self.data]

  @property
  def provide_label(self):
  """The name and shape of label provided by this iterator"""
  return [(k, tuple([self.batch_size] + list(v.shape[1: ] ) )) for k, v in self.label]


  def hard_reset(self):
  """Igore roll over data and set to start"""
  self.cursor = -self.batch_size

  def reset(self):
    if self.last_batch_handle == 'roll_over' and self.cursor > self.num_data:
      self.cursor = -self.batch_size + (self.cursor % self.num_data) % self.batch_size
    else:
      self.cursor = -self.batch_size

  def iter_next(self):
    self.cursor += self.batch_size
    if self.cursor < self.num_data:
      return True
    else:
      return False

  def next(self):
    if self.iter_next():
      return DataBatch(data = self.getdata(), label = self.getlabel(), \
        pad = self.getpad(), index = None)
    else:
      raise StopIteration

  def _getdata(self, data_source):
    """Load data from underlying arrays, internal use only"""
    assert(self.cursor < self.num_data), "DataIter needs reset."
    if self.cursor + self.batch_size <= self.num_data:
      return[array(x[ 1][self.cursor: self.cursor + self.batch_size] ) for x in data_source]
    else:
      pad = self.batch_size - self.num_data + self.cursor
      return[array(np.concatenate((x[ 1][self.cursor:], x[ 1][: pad] ), axis = 0) ) for x in data_source]

  def getdata(self):
    return self._getdata(self.data)

  def getlabel(self):
    return self._getdata(self.label)

  def getpad(self):

  if self.last_batch_handle == 'pad ' and \
    self.cursor + self.batch_size > self.num_data:
    return self.cursor + self.batch_size - self.num_data
  else:
    return 0
}
*/
