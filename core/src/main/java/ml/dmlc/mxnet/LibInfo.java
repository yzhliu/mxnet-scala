package ml.dmlc.mxnet;

import ml.dmlc.mxnet.Base.*;

class LibInfo {
  public native int mxNDArrayFree(NDArrayHandle handle);
  public native String mxGetLastError();
  public native int mxNDArrayCreateNone(NDArrayHandle out);
}
