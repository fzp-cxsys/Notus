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

    public static boolean transformData2CVS(INDArray data, String fileName) throws URISyntaxException, IOException {
        File root = new File(Data2CVS.class.getClassLoader().getResource("data/csv").toURI());
        BufferedWriter writer = new BufferedWriter(new FileWriter(new File(root, fileName), true));

        for (int i = 0; i < data.rows(); i++) {
            StringBuffer s = new StringBuffer();
            INDArray row = data.getRow(i);
            for (int j = 0; j < row.columns(); j++) {
                if(j != row.columns() - 2 && j != row.columns() - 3 && j != row.columns() - 4){
                    double aDouble = row.getDouble(j);
                    s.append(aDouble);
                    s.append(',');
                }
            }
            s.append(row.getInt(row.columns() - 4));
            s.append('-');
            s.append(row.getInt(row.columns() - 3));
            s.append('-');
            s.append(row.getInt(row.columns() - 2));
            writer.write(s.toString());
            writer.newLine();
        }

        writer.close();

        return true;
    }

    public static boolean transform(String toFileName) throws IOException, URISyntaxException {
        for (INDArray data: dp) {
            if(!transformData2CVS(data, toFileName)) return false;
        }
        return true;
    }

    public static void main(String[] args) throws IOException, URISyntaxException {
        transform("data0");
    }
}
