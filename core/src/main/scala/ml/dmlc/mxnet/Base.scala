package ml.dmlc.mxnet

class NDArrayHandle

object Base {
  // TODO
  System.loadLibrary("mxnet-scala")
  val _LIB = new LibInfo


  // helper function definition
  def check_call(ret: Int): Unit = {
    /**
      Check the return value of C API call

      This function will raise exception when error occurs.
      Wrap every API call with this function

      Parameters
      ----------
      ret : int
          return value from API calls
    */
    if (ret != 0) {
      throw new MXNetError(_LIB.mxGetLastError())
    }
  }
}

class MXNetError(val err: String) extends Exception(err)
