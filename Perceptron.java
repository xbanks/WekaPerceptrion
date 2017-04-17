/**
 *  Xavier Banks
 *  CAP4630 Fall 2016
 *  Assignment 3
 *  Perceptron Classifier
 */


import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;

public class Perceptron implements weka.classifiers.Classifier {

    private final String filename;
    private final int numEpochs;
    private final double learningRate;
    private final ArrayList<Double> weights;
    private int weightUpdates = 0;

    public Perceptron(String[] options) {
        filename = options[0];
        numEpochs = Integer.parseInt(options[1]);
        learningRate = Double.parseDouble(options[2]);
        weights = new ArrayList<>();
    }

    private void printHeading() {
        System.out.println("University of Central Florida");
        System.out.println("CAP4630 Artificial Intelligence - Fall 2016");
        System.out.println("Perceptron Classifier by Xavier Banks\n");
    }

    @Override
    public void buildClassifier(Instances instances) throws Exception {
        printHeading();

        int numFeatures = instances.numAttributes() - 1;
        initWeights(numFeatures);


        for (int epoch = 0; epoch < numEpochs; epoch++) {
            String epochResult = "";
            for (Instance instance : instances) {
                final double actualClass = instance.classValue();
                ArrayList<Double> features = getFeatures(instance);

                double prediction = Predict(features);

                updateWeights(features, prediction, actualClass);
                epochResult += prediction == actualClass ? "1" : "0";
            }

            System.out.printf("Epoch\t%4d: %s\n", epoch, epochResult);
        }
    }

    private ArrayList<Double> getFeatures(Instance instance) {
        ArrayList<Double> features = new ArrayList<>();
        int numFeatures = instance.numAttributes() - 1; // -1 because we're leaving out the class attribute
        for (int i = 0; i < numFeatures; i++) {
            Attribute attribute = instance.attribute(i);
            features.add(instance.value(attribute));
        }

        return features;
    }

    private double Predict(Instance instance) {
        return Predict(getFeatures(instance));
    }

    private double Predict(ArrayList<Double> features) {
        ArrayList<Double> biasedFeatures = applyBias(features);
        final double actLevel = activationLevel(biasedFeatures);

        return threshold(actLevel);
    }

    private void initWeights(int numFeatures) {
        if (!weights.isEmpty()) return; // Weights have already been initialized
        for (int i = 0; i < numFeatures + 1; i++) {
            // All weights start at 0
            weights.add(0.0);
        }
    }

    private ArrayList<Double> applyBias(ArrayList<Double> features) {
        ArrayList<Double> biasedFeatures = new ArrayList<>();
        biasedFeatures.addAll(features);
        biasedFeatures.add(1.0); // Add the bias value to the END of the list
        return biasedFeatures;
    }

    private ArrayList<Double> weighDown(ArrayList<Double> features) {
        ArrayList<Double> weighted = new ArrayList<>();
        for (int i = 0; i < features.size(); i++) {
            // Apply the weight to each attribute value
            final double w = features.get(i) * weights.get(i);
            weighted.add(w);
        }

        return weighted;
    }

    private double activationLevel(ArrayList<Double> features) {
        ArrayList<Double> weightedValues = weighDown(features);
        return weightedValues.stream().reduce(0.0, (a, b) -> a + b);
    }

    private double threshold(double activationLevel) {
        return activationLevel >= 0 ? 0 : 1; // 0 == 'a', 1 == 'b'
    }

    private void updateWeights(ArrayList<Double> features, double predictedClass, double actualClass) {
        final double diff = actualClass - predictedClass;
        if (diff == 0) return; // Don't change the weights if the classification was correct

        ArrayList<Double> biasedFeatures = applyBias(features);
        for (int i = 0; i < biasedFeatures.size(); i++) {
            // wi = wi + c(y - h(xi))xi
            final double w_old = weights.get(i);
            final double feature = biasedFeatures.get(i);
            final double delta_w = -2.0 * diff * learningRate * feature;
            final double w_new = w_old + delta_w;
            weights.set(i, w_new);
        }

        weightUpdates++;
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        return Predict(instance);
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        final double prediction = Predict(instance);
        return (prediction == 0) ? new double[]{1.0, 0.0} : new double[]{0.0, 1.0};
    }

    @Override
    public Capabilities getCapabilities() {
        return null;
    }

    @Override
    public String toString() {
        String str = String.format("Source File: %s\nTraining Epochs: %d\nLearning rate: %.2f\n\n",
                filename, numEpochs, learningRate);

        str += String.format("Total # weight updates = %d\n", weightUpdates);
        str += "Final weights:\n";
        for (double weight : weights) {
            str += String.format("%6.3f\n", weight);
        }

        return str;
    }
}