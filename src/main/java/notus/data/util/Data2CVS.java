package notus.data.util;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.net.URISyntaxException;

/**
 * Created by a1477 on 2017/3/3.
 */
public class Data2CVS {
    private static DataProcessor dp = null;
    static {
        try {
            dp = new DataProcessor();
        } catch (URISyntaxException e) {
            throw new RuntimeException(e);
        }
    }

    public static boolean transformData2CVS(INDArray data) throws URISyntaxException, IOException {

        File root = new File(Data2CVS.class.getClassLoader().getResource("data/cvs").toURI());
        BufferedWriter writer = new BufferedWriter(new FileWriter(new File(root, "hh")));
        return true;
    }

    public static boolean transform() {
        for (INDArray data: dp) {
//            if(!transformData2CVS(data)) return false;
        }
        return true;
    }
}
