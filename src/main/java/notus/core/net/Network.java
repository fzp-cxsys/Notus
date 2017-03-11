package notus.core.net;

import notus.data.util.DataProcessor;
import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.net.URISyntaxException;

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
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
                .list()
                .layer(0, new RBM.Builder().nIn(17).nOut(50).lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE).build())
                .layer(1, new RBM.Builder().nIn(50).nOut(30).lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE).build())
                .layer(2, new RBM.Builder().nIn(30).nOut(10).lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE).build())
                .layer(3, new DenseLayer.Builder().activation(Activation.SIGMOID).nIn(10).nOut(8).build())
                .layer(4, new OutputLayer.Builder(LossFunctions.LossFunction.MSE).nIn(8).nOut(1).build())
                .pretrain(true).backprop(true)
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        model.setListeners(new ScoreIterationListener(listenerFreq));

        log.info("Train model....");
        DataProcessor dataProcessor = new DataProcessor();
        INDArray data = dataProcessor.nextData();
        DataSet dataSet = new DataSet(data.getColumns(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17), data.getColumns(21));

//        new DataProcessor().nextData()

        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(dataSet);
        normalizer.transform(dataSet);

        System.out.println(dataSet.getLabels());
        model.fit(dataSet);

        System.out.println(model.output(data.getRow(0).getColumns(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17)));

    }
}
