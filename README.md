# Thema09-2023

# WekaRunner

## Introduction
Welcome to the WekaRunner project! This Java application utilizes the Weka library for machine learning to classify patients into "Demented" or "Non-demented" categories based on MMSE scores. The project consists of a command-line interface for both analyzing datasets and making predictions for single instances.

## Prerequisites
Make sure you have the following installed on your system:
- Java (JRE or JDK)
- Weka library

## Usage

### Analyzing a Dataset
To analyze a dataset and get the number of "Demented" and "Non-demented" cases, use the following command:

```bash
java -jar WekaAplicatie-1.0-SNAPSHOT-all.jar -f <path-to-dataset-file>
```

Ensure that the dataset file has a '.arff' extension.

### Making Predictions for Single Instance
To make predictions for a single instance based on the MMSE score, use the following command:

```bash
java -jar WekaAplicatie-1.0-SNAPSHOT-all.jar -v <MMSE-score>
```

Make sure to replace `<MMSE-score>` with the actual MMSE score you want to predict.

## Options

- **-f, --file**: Analyze a dataset. Provide the path to the dataset file.

- **-v, --value**: Make predictions for a single instance. Provide the MMSE score.

## Examples

### Analyzing a Dataset

```bash
java -jar WekaAplicatie-1.0-SNAPSHOT-all.jar -f DataGuess.arff
```

### Making Predictions for Single Instance

```bash
java -jar WekaAplicatie-1.0-SNAPSHOT-all.jar -v 28
```

## Important Notes
- Ensure that the dataset follows the '.arff' format.
- The MMSE score must be a numeric value between 1 and 30.
