from libs.TrashbinDataset import TrashbinDataset ## run local
# from TrashbinDataset import TrashbinDataset ## run colab
from torch.utils.data import DataLoader

class TDContainer:
    """ Class that contains the dataset for training, validation and test
        Attributes
        ----------
        self.training, self.validation, self.test are the TrashbinDataset object
        self.training_loader, self.validation_loader, self.test_loader are DataLoader of the correspective TrashbinDataset
    """

    def __init__(self, training=None, validation=None, test=None):
        """ Constructor of the class. Instantiate an TrashbinDataset dataset for each dataset

            Parameters
            ----------
            training: str, required
                path of training dataset csv
            
            validation: str, required
                path of validation dataset csv
            
            test: str, required
                path of test dataset csv
        """
    
        if training is None or validation is None or test is None:
            raise NotImplementedError("No default dataset is provided")
        
        if isinstance(training, dict) is False or isinstance(validation, dict) is False or isinstance(test, dict) is False:
            raise NotImplementedError("Constructor accept only dict file.")

        if training['path'] is None or validation['path'] is None or test['path'] is None or isinstance(training['path'], str) is False or isinstance(validation['path'], str) is False or isinstance(test['path'], str) is False:
            raise NotImplementedError("Path file is required and need to be a str type.")

        self.training = TrashbinDataset(training['path'], transform=training['transform'])
        self.validation = TrashbinDataset(validation['path'], transform=validation['transform'])
        self.test = TrashbinDataset(test['path'], transform=test['transform'])
        self.hasDl = False

    def create_data_loader(self, batch_size=32, num_workers=2, drop_last=False):
        """ Create data loader for each dataset https://pytorch.org/docs/stable/data.html """

        if isinstance(batch_size, int) is False or isinstance(num_workers, int) is False:
            raise NotImplementedError("Parameters accept only int value.")

        self.training_loader = DataLoader(self.training, batch_size=batch_size, num_workers=num_workers, drop_last=drop_last, shuffle=True)
        self.validation_loader = DataLoader(self.validation, batch_size=batch_size, num_workers=num_workers, drop_last=drop_last)
        self.test_loader = DataLoader(self.test, batch_size=batch_size, num_workers=num_workers, drop_last=drop_last)
        self.hasDl = True

    def show_info(self):
        """ Print info of dataset """
        print("\n=== *** DB INFO *** ===")
        print("Training:", self.training.__len__(), "values, \nValidation:", self.validation.__len__(), "values, \nTest:", self.test.__len__())
        print("DataLoader:", self.hasDl)
        print("=== *** END *** ====\n")