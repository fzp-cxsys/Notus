package notus.data.util;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;
import java.net.URISyntaxException;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by a1477 on 2017/3/1.
 */
public class DataProcessor{
    private URL basePath;
    private File[] files = null;
    private int index = 0;

    public DataProcessor(){
        this.basePath = DataProcessor.class.getClassLoader().getResource("data");
    }

    public INDArray nextInput() throws URISyntaxException, IOException {
        if (files == null){ //  数据文件列表为空时先设置文件列表
            File root = new File(this.basePath.toURI());
            this.files = root.listFiles();
        }

        //  读取当前遍历到的文件
        BufferedReader reader = new BufferedReader(new FileReader(this.files[index++]));
        //  读取文件头信息
        String line1 = reader.readLine();
        String[] l1splits = line1.split(" ");
        //  记录降雨量数据的index
        int rainfallIndex = 0;
        for (int i = 0; i < l1splits.length; i++) {
            if(l1splits[i].equals("V13201")){
                rainfallIndex = i;
            }
        }

        //  将数据文件内容读取到内存
        List<String> dataContent = new ArrayList<>();
        String newLine = reader.readLine();
        while (newLine != null){
            dataContent.add(newLine);
            newLine = reader.readLine();
        }

        //  初始一个矩阵
        INDArray data = Nd4j.create(dataContent.size(), l1splits.length);
        for (int i = 0; i < dataContent.size(); i++) {
            //  处理一行数据，将降雨量数据放在最后一位
            double[] vector = new double[l1splits.length];
            String[] split = dataContent.get(i).split(" ");
            int k = 0;
            for (int j = 0; j < split.length; j++) {
                if (j != rainfallIndex){
                    vector[k++] = Double.parseDouble(split[j]);
                } else {
                    vector[vector.length - 1] = Double.parseDouble(split[j]);
                }
            }
            //  将这行数据置入矩阵
            data.putRow(i, Nd4j.create(vector));
        }

        return data;
    }

    public static void main(String[] args) throws URISyntaxException, IOException {
        DataProcessor dataProcessor = new DataProcessor();
        INDArray input = dataProcessor.nextInput();
        System.out.println(input);
    }
}
