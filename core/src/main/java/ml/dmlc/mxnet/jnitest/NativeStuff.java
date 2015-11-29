package ml.dmlc.mxnet.jnitest;

public class NativeStuff {
  public native void helloNative();
  public static void main(String[] args) {
    System.loadLibrary("mxnet-scala");
    new NativeStuff().helloNative();
  }
}
