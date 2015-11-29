package ml.dmlc.mxnet.jnitest;

public class NativeStuff {
  public native void helloNative();
  public static void main(String[] args) {
    //System.load("/Users/lewis/Workspace/ml/mxnet-scala/native/osx-x86_64/target/mxnet-scala-native.so");
    System.loadLibrary("mxnet-scala");
    new NativeStuff().helloNative();
  }
}
