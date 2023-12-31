import torch
import config
import argparse
from Source.data import TextDataset
from Source.utils import load_file, save_file
from sklearn.model_selection import train_test_split
from Source.model import RNNNetwork, LSTMNetwork, train, test

# Define the main function that orchestrates the training and testing process
def main(args_):

    print("Loading the files...")
    # Load preprocessed data and embeddings
    tokens = load_file(config.tokens_path)
    labels = load_file(config.labels_path)
    embeddings = load_file(config.embeddings_path)
    label_encoder = load_file(config.label_encoder_path)
    num_classes = len(label_encoder.classes_)

    print("Splitting data into train, valid, and test sets...")
    # Split the data into training, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(tokens, labels, test_size=0.2)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.25)

    print("Creating PyTorch datasets...")
    # Create PyTorch datasets for training, validation, and testing
    train_dataset = TextDataset(X_train, embeddings, y_train)
    valid_dataset = TextDataset(X_valid, embeddings, y_valid)
    test_dataset = TextDataset(X_test, embeddings, y_test)

    print("Creating data loaders...")
    # Create data loaders for training, validation, and testing
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=16)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16)

    print("Creating model object...")
    # Create an instance of the selected neural network model (RNN or LSTM)
    if args_.model_type == "lstm":
        model = LSTMNetwork(config.input_size, config.hidden_size, num_classes)
        model_path = config.lstm_model_path
    else:
        model = RNNNetwork(config.input_size, config.hidden_size, num_classes)
        model_path = config.rnn_model_path

    # Move the model to GPU if a GPU is available
    if torch.cuda.is_available():
        model = model.cuda()

    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Training the model...")
    # Train the model using the training and validation data
    train(train_loader, valid_loader, model, criterion, optimizer, device, config.num_epochs, model_path)

    print("Testing the model...")
    # Test the trained model on the test data
    test(test_loader, model, criterion, device)

if __name__ == "__main__":
    # Parse command-line arguments for selecting the model type
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default='rnn', help="Model type: lstm or rnn")
    args = parser.parse_args()
    main(args)
