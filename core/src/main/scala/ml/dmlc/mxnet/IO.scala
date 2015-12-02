package ml.dmlc.mxnet

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
