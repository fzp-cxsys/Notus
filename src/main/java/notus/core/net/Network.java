package notus.core.net;

import notus.data.util.DataProcessor;
import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by a1477 on 2017/3/1.
 */
public class Network {
    private static Logger log = LoggerFactory.getLogger(Network.class);

    public static void main(String[] args) throws IOException, URISyntaxException {
        int seed = 123;
        int iterations = 1;
        int listenerFreq = iterations/5;


        log.info("Build model....");
//        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
//                .seed(seed)
//                .iterations(iterations)
//                .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
//                .list()
//                .layer(0, new RBM.Builder().nIn(17).nOut(50).lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE).build())
//                .layer(1, new RBM.Builder().nIn(50).nOut(30).lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE).build())
//                .layer(2, new RBM.Builder().nIn(30).nOut(10).lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE).build())
//                .layer(3, new DenseLayer.Builder().activation(Activation.SIGMOID).nIn(10).nOut(10).build())
//                .layer(4, new OutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.SOFTMAX).nIn(10).nOut(8).build())
//                .pretrain(true).backprop(true)
//                .build();
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed) //include a random seed for reproducibility
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT) // use stochastic gradient descent as an optimization algorithm
                .iterations(iterations)
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .learningRate(0.0015) //specify the learning rate
                .updater(Updater.NESTEROVS).momentum(0.98) //specify the rate of change of the learning rate.
                .regularization(true).l2(0.0015 * 0.005) // regularize learning model
                .list()
                .layer(0, new DenseLayer.Builder() //create the first input layer.
                        .nIn(17)
                        .nOut(14)
                        .build())
                .layer(1, new DenseLayer.Builder() //create the second input layer
                        .nIn(14)
                        .nOut(10)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD) //create hidden layer
                        .activation(Activation.SOFTMAX)
                        .nIn(10)
                        .nOut(8)
                        .build())
                .pretrain(false).backprop(true) //use backpropagation to adjust weights
                .build();


        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        model.setListeners(new ScoreIterationListener(listenerFreq));

        log.info("Train model....");
        DataProcessor dataProcessor = new DataProcessor();
        INDArray testData = dataProcessor.nextData();
        for (INDArray data : dataProcessor) {
            data = dataProcessor.nextData();
            INDArray orinLabels = data.getColumns(21);
            INDArray labels = Nd4j.zeros(data.rows(), 8);

//    小雨：1d（或24h）降雨量小于10mm者。
//　　中雨：1d（或24h）降雨量10～25mm者。
//　　大雨：1d（或24h）降雨量25～50mm者。
//　　暴雨：1d（或24h）降雨量50～100mm者。
//　　大暴雨：1d（或24h）降雨量100～200mm者。
//　　特大暴雨：1d（或24h）降雨量在200mm以上者。
            for (int i = 0; i < data.rows(); i++) {
                double rainfall = orinLabels.getDouble(i);
                if (rainfall == 0) {
                    labels.put(i, 0, 1);
                } else if(rainfall >= 30000) {
                    labels.put(i, 1, 1);
                } else if (rainfall < 10) {
                    labels.put(i, 2, 1);
                } else if (rainfall >= 10 && rainfall < 25) {
                    labels.put(i, 3, 1);
                } else if (rainfall >= 10 && rainfall < 25) {
                    labels.put(i, 4, 1);
                } else if (rainfall >= 25 && rainfall < 50) {
                    labels.put(i, 5, 1);
                } else if (rainfall >= 50 && rainfall < 100) {
                    labels.put(i, 6, 1);
                } else {
                    labels.put(i, 7, 1);
                }
            }
//        System.out.println(labels);
            DataSet dataSet = new DataSet(data.getColumns(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17), labels);


            List<String> labelNames = Arrays.asList("微雨", "小雨", "中雨" , "大雨", "暴雨", "大暴雨", "特大暴雨");

            dataSet.setLabelNames(labelNames);

            DataNormalization normalizer = new NormalizerStandardize();
            normalizer.fit(dataSet);
            normalizer.transform(dataSet);

//            System.out.println(dataSet.getLabels());
            model.fit(dataSet);

        }
//        System.out.println(model.predict(dataSet));
        System.out.println(Arrays.toString(model.predict(testData.getColumns(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17))));
        System.out.println(model.output(testData.getColumns(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17)));

    }
}
