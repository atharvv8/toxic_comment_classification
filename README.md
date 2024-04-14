# Toxic Comment Classification Project

This project aims to classify toxic comments using natural language processing and machine learning techniques. It uses a dataset of comments labeled with different categories of toxicity and employs a neural network model to predict the categories of toxicity present in new comments.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Model Saving](#model-saving)
- [Visualization](#visualization)
- [Docker](#docker)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/atharvv8/toxic_comment_classification
    cd yourrepository
    ```

2. Install the required dependencies using the provided `requirements.txt`:

    ```bash
    pip install -r requirements.txt
    ```

The dependencies include:

- TensorFlow
- Pandas
- NumPy

These libraries are used for model training, data manipulation, and other computations.

## Usage

The project provides a trained model for toxicity classification in text comments. To use the model, follow these steps:

1. Run the script:

    ```bash
    python Comments_Toxicity.py
    ```

2. Input the comment you want to classify, and the script will display the predicted categories of toxicity present in the comment.

## Model Training

The training process includes data preprocessing, text vectorization, and neural network model training. The script uses the following steps:

1. Load the dataset from CSV files.
2. Preprocess the data and split it into training and validation sets.
3. Train a neural network model using the preprocessed data.
4. Use callbacks to monitor and adjust the learning rate during training.

## Model Evaluation

The model is evaluated on a separate test dataset and provides the loss and binary accuracy metrics.

## Model Saving

The trained model is saved using TensorFlow's `model.save()` method in the script. The saved model can be used for predictions on new data.

## Visualization

The script includes a visualization of the training and validation accuracy over epochs. This helps to assess the model's performance and convergence during training.

## Docker

You can also run the project inside a Docker container. A Dockerfile is provided in the repository for this purpose.

The Dockerfile uses the following specifications:

- `FROM python:3.9`
- Sets the base image to Python 3.9.

- `WORKDIR /jigsaw-toxic-comment-classification-challenge`
- Sets the working directory inside the container.

- `COPY requirements.txt .`
- Copies the `requirements.txt` file into the working directory.

- `RUN pip install --upgrade pip`
- Upgrades `pip` to the latest version.

- `RUN pip install --no-cache-dir -r requirements.txt`
- Installs the project dependencies listed in the `requirements.txt` file.

- `COPY . .`
- Copies all files and directories from the current directory to the working directory inside the container.

- `CMD ["python3","./Comments_Toxicity.py"]`
- Sets the default command to execute the Python script that runs the toxicity classification model.

To build the Docker image, use the following command in your terminal:

```bash
docker build -t toxic-comment-classification .
