package nl.bioinf;

import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import org.apache.commons.cli.*;

public class WekaRunner {

    private static final Option ARG_file = new Option("f", "file", false, "Looks through a file and gives the number of demented and non-demented cases");
    private static final Option ARG_value = new Option("v", "value", false, "Pulls the given value of a patient through the model and gives a conclusion");
    private final String modelFile = "J48FinalModel.model";

    // Method to print help information
    private static void printHelp(Options options) {
        HelpFormatter formatter = new HelpFormatter();
        PrintWriter pw = new PrintWriter(System.out);
        pw.println("Dementia checker " + Math.class.getPackage().getSpecificationVersion());
        pw.println();
        formatter.printUsage(pw, 100, "java -jar nl.bioinf.WekaRunner.jar [options for testing] File/Value");
        formatter.printOptions(pw, 100, options, 2, 5);
        pw.close();
    }

    // Main method
    public static void main(String[] args) {
        CommandLineParser clp = new DefaultParser();
        Options options = new Options();
        options.addOption(ARG_file);
        options.addOption(ARG_value);

        try {
            CommandLine cl = clp.parse(options, args);
            if (cl.getArgList().size() < 1) {
                System.out.println("\nWelcome to this dementia app checker.\n" +
                        "It seems you did not put any file/value or chose an option to start the app.\n" +
                        "Look at the usage below on how to start the app\n");
                printHelp(options);
                System.exit(-1);
            }
            if (cl.hasOption(ARG_file.getLongOpt())) {
                try {
                    String dataFilePath = args[1];
                    if ((dataFilePath.endsWith(".arff"))) {
                        System.out.println("Begin");
                        WekaRunner runner = new WekaRunner();
                        runner.start(dataFilePath);
                    } else {
                        System.out.println("\nSomething went wrong. Please check the following:\n" +
                                "1. The file must have a .arff extension.\n" +
                                "2. It must be a valid file.\n" +
                                "3. Use the option before the file.\n");
                        printHelp(options);
                    }
                } catch (Exception e) {
                    printHelp(options);
                    e.printStackTrace();
                    System.exit(-1);
                }
            } else if (cl.hasOption(ARG_value.getLongOpt())) {
                int MMSE;
                try {
                    MMSE = Integer.parseInt(cl.getArgList().get(0));
                    WekaRunner runner = new WekaRunner();
                    runner.start2(MMSE);
                } catch (Exception e) {
                    System.out.println("\nSomething went wrong with the value you entered. Please pay attention to the following:\n" +
                            "1. Did you provide a number between 1 and 30?\n" +
                            "2. Did you enter a letter instead of a number?\n");
                    printHelp(options);
                    e.printStackTrace();
                    System.exit(-1);
                }
            }
        } catch (Exception e) {
            printHelp(options);
            e.printStackTrace();
        }
    }

    // Method to start processing a file
    private void start(String dataFilePath) {
        Options options = new Options();
        options.addOption(ARG_file);
        options.addOption(ARG_value);

        try {
            String dataFile = "FinalData.arff";
            Instances instances = loadArff(dataFile);
            printInstances(instances);
            J48 j48 = buildClassifier(instances);
            saveClassifier(j48);
            J48 fromFile = loadClassifier();
            Instances unknownInstances = loadArff(dataFilePath);
            System.out.println("\nUnclassified unknownInstances = \n" + unknownInstances);
            classifyNewInstance(fromFile, unknownInstances);

        } catch (Exception e) {
            e.printStackTrace();
            printHelp(options);
        }
    }
    // Method to start processing a value
    private void start2(int MMSE) {
        Options options = new Options();
        options.addOption(ARG_file);
        options.addOption(ARG_value);

        try {
            String dataFile = "FinalData.arff";
            Instances instances = loadArff(dataFile);
            printInstances(instances);
            J48 j48 = buildClassifier(instances);
            saveClassifier(j48);
            J48 fromFile = loadClassifier();

            Instances dataset = createDatasetWithNewAttribute(MMSE);

            System.out.println("\nUnclassified unknownInstances = \n" + dataset);

            classifyNewInstance(fromFile, dataset);

        } catch (Exception e) {
            e.printStackTrace();
            printHelp(options);
        }
    }
    // Method to classify a new instance
    private void classifyNewInstance(J48 tree, Instances unknownInstances) throws Exception {
        // create copy
        Instances labeled = new Instances(unknownInstances);
        // label instances
        for (int i = 0; i < unknownInstances.numInstances(); i++) {
            double clsLabel = tree.classifyInstance(unknownInstances.instance(i));
            labeled.instance(i).setClassValue(clsLabel);
        }
        System.out.println("\nNew, labeled = \n" + labeled);
    }

    // Method to load a classifier from a file
    private J48 loadClassifier() throws Exception {
        // deserialize model
        return (J48) weka.core.SerializationHelper.read(modelFile);
    }

    // Method to save a classifier to a file
    private void saveClassifier(J48 j48) throws Exception {
        // serialize model
        weka.core.SerializationHelper.write(modelFile, j48);
    }

    // Method to build a classifier
    private J48 buildClassifier(Instances instances) throws Exception {
        String[] options = new String[1];
        options[0] = "-U"; // un-pruned tree
        J48 tree = new J48(); // new instance of tree
        tree.setOptions(options); // set the options
        tree.buildClassifier(instances); // build classifier
        return tree;
    }

    // Method to print information about instances
    private static void printInstances(Instances instances) {
        int numAttributes = instances.numAttributes();

        for (int i = 0; i < numAttributes; i++) {
            System.out.println("attribute " + i + " = " + instances.attribute(i));
        }
        System.out.println("class index = " + instances.classIndex());
    }

    // Method to load ARFF file
    private static Instances loadArff(String datafile) throws IOException {
        Options options = new Options();
        options.addOption(ARG_file);
        options.addOption(ARG_value);
        try {
            DataSource source = new DataSource(datafile);
            Instances data = source.getDataSet();
            // setting class attribute if the data format does not provide this information
            // For example, the ARFF format saves the class attribute information as well
            if (data.classIndex() == -1)
                data.setClassIndex(data.numAttributes() - 1);
            return data;
        } catch (Exception e) {
            printHelp(options);
            throw new IOException("\n\nThe file you provided could not be read. Please provide a correct file.\n");
        }
    }
    // Method to create a dataset with a new attribute (MMSE score)
    private static Instances createDatasetWithNewAttribute(int Value) {
        // Define the attribute for the MMSE score
        Attribute MSSE = new Attribute("MMSE");

        // Define the attribute for the class label ("Demented" or "Non-demented")
        ArrayList<String> classValues = new ArrayList<>();
        classValues.add("Non-demented");
        classValues.add("Demented");
        Attribute classAttribute = new Attribute("Class", classValues);

        // Create an Instances object with both attributes
        ArrayList<Attribute> attributes = new ArrayList<>();
        attributes.add(MSSE);
        attributes.add(classAttribute);

        Instances dataset = new Instances("MyDataset", attributes, 0);

        // Create a new instance and set the MMSE score and "?" for class label
        Instance single = new DenseInstance(2); // Two attributes
        single.setValue(MSSE, Value);
        single.setMissing(classAttribute);

        // Add the instance to the dataset
        dataset.add(single);
        if (dataset.classIndex() == -1)
            dataset.setClassIndex(dataset.numAttributes() - 1);
        return dataset;
    }
}
