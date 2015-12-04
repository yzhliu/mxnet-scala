package ml.dmlc.mxnet;

class LibInfo {
  public native int mxNDArrayFree(NDArrayHandle handle);
  public native String mxGetLastError();
}
