from abc import ABC, abstractmethod

class BaseDetector(ABC):
    @abstractmethod
    def train(self, dataloader, epochs):
        """
        Train the detector using the provided dataloader for a given number of epochs.

        Pareameters:
        dataloader: DataLoader object that provides the training data.
        epochs: int, number of epochs to train the model.
        """
        pass

    @abstractmethod
    def evaluate(self, dataloader):
        """
        Evaluate the model using the provided dataloader.

        Parameters:
        dataloader (DataLoader): The dataloader providing the evaluation data.
        """
        pass

    @abstractmethod
    def inference(self, inputs):
        """
        Perform inference on the given inputs.

        Parameters:
        inputs: The input data for which inference needs to be performed.
        """
        pass