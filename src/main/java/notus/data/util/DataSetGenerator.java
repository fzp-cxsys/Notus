package notus.data.util;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;

import java.io.FileNotFoundException;
import java.io.IOException;

/**
 * Created by a1477 on 2017/3/6.
 */
public class DataSetGenerator {
    public static void main(String[] args) throws IOException, InterruptedException {
        RecordReader recordReader = new CSVRecordReader();
        recordReader.initialize(new FileSplit(new ClassPathResource("data/csv/data0").getFile()));
//        new RecordReaderDataSetIterator(recordReader, 100, 22, );
        System.out.println();
    }
}
