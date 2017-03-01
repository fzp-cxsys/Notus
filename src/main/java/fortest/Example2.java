package fortest;

import org.nd4j.linalg.api.buffer.BaseDataBuffer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DoubleBuffer;
import org.nd4j.linalg.api.ndarray.BaseNDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;

import java.io.BufferedInputStream;
import java.io.FileInputStream;

/**
 * Created by a1477 on 2017/2/28.
 */
public class Example2 {
    public static void main(String[] args) throws Exception {
        BufferedInputStream in = new BufferedInputStream(new FileInputStream("result"));
        NDArray data = new NDArray(new DoubleBuffer(100), new int[]{1 , 2882028});
        data.data().read(in);
        System.out.println(data.getColumns(1,2,3));
    }
}
