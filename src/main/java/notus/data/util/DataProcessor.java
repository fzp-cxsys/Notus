package notus.data.util;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;
import java.net.URISyntaxException;
import java.net.URL;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * Created by a1477 on 2017/3/1.
 */
public class DataProcessor implements Iterable<INDArray>{
    private URL basePath;
    private File[] files = null;
    private int index = 0;

    public DataProcessor() throws URISyntaxException {
        this.basePath = DataProcessor.class.getClassLoader().getResource("data/original");
        File root = new File(this.basePath.toURI());
        this.files = root.listFiles();
    }

    private INDArray transformDate (List<String> dataContent, final int dimension, final int rainfallIndex, final int year, final int month, final int day){
        //  初始一个矩阵
        INDArray data = Nd4j.create(dataContent.size(), dimension);
        for (int i = 0; i < dataContent.size(); i++) {
            //  处理一行数据，将降雨量数据放在最后一位
            double[] vector = new double[dimension];
            String[] split = dataContent.get(i).split(" ");
            int k = 0;
            for (int j = 0; j < split.length; j++) {
                if (j == rainfallIndex){
                    vector[vector.length - 1] = Double.parseDouble(split[j]);
                } else if (j == year) {
                    vector[vector.length - 4] = Double.parseDouble(split[j]);
                } else if (j == month) {
                    vector[vector.length - 3] = Double.parseDouble(split[j]);
                } else if (j == day) {
                    vector[vector.length - 2] = Double.parseDouble(split[j]);
                } else {
                    vector[k++] = Double.parseDouble(split[j]);
                }
            }
            //  将这行数据置入矩阵
            data.putRow(i, Nd4j.create(vector));
        }
        return data;
    }

    public INDArray nextData() throws URISyntaxException, IOException {
        if (this.index == this.files.length) return null;

        //  读取当前遍历到的文件
        BufferedReader reader = new BufferedReader(new FileReader(this.files[index++]));
        //  读取文件头信息
        String line1 = reader.readLine();
        String[] l1splits = line1.split(" ");
        //  记录降雨量数据和日期的index
        int rainfallIndex = 0;
        int year = 0;
        int month = 0;
        int day = 0;
        for (int i = 0; i < l1splits.length; i++) {
            switch (l1splits[i]) {
                case "V13201" : rainfallIndex = i;break;
                case "V04001" : year = i;break;
                case "V04002" : month = i;break;
                case "V04003" : day = i;break;
                default : break;
            }
        }

        //  将数据文件内容读取到内存
        List<String> dataContent = new ArrayList<>();
        String newLine = reader.readLine();
        while (newLine != null){
            dataContent.add(newLine);
            newLine = reader.readLine();
        }

        return transformDate(dataContent, l1splits.length, rainfallIndex, year, month, day);
    }

    @Override
    public Iterator<INDArray> iterator() {
        return new Iterator<INDArray>() {
            @Override
            public boolean hasNext() {
                if (files != null && index < files.length){
                    return true;
                } else {
                    return false;
                }
            }

            @Override
            public INDArray next() {
                try {
                    return nextData();
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            }
        };
    }

    public static void main(String[] args) throws URISyntaxException, IOException {
        DataProcessor dataProcessor = new DataProcessor();
        for (INDArray data:dataProcessor) {
            System.out.println(data);
        }
    }
}
