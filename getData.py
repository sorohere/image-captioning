import tensorflow as tf
import pathlib
import shutil
import os

class Flickr8kDatasetHandler:
    """
    A class to handle the downloading, extraction, and organization of the Flickr8k dataset and its text files.
    
    Methods:
        get_data(path: str) -> None:
            Downloads and extracts the Flickr8k dataset and text files.
        
        check_and_move_files(path: str) -> None:
            Checks if the required files and directories exist. If not, moves the files and directories
            from the extracted zip directories to the correct locations.
    """

    def __init__(self, path='./dataset/flickr8k'):
        """
        Initializes the dataset handler with the given dataset path.
        
        Parameters:
            path (str): Path where the dataset and text files will be stored. Default is './dataset/flickr8k'.
        """
        self.path = pathlib.Path(path)
    
    def get_data(self):
        """
        Downloads and extracts the Flickr8k dataset and text files.
        
        If the dataset has already been downloaded and extracted, this method does nothing.
        Otherwise, it downloads the dataset and text files from the specified URLs and extracts them.
        """
        dataset_path = self.path / 'Flicker8k_Dataset'  # Directory created after dataset extraction
        print()
        # Check if dataset and text files are already extracted
        if not dataset_path.exists():
            # Download and extract the Flickr8k Dataset
            tf.keras.utils.get_file(
                origin='https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip',
                cache_dir='.',
                cache_subdir=self.path,
                extract=True
            )
            
            # Download and extract the Flickr8k text files
            tf.keras.utils.get_file(
                origin='https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip',
                cache_dir='.',
                cache_subdir=self.path,
                extract=True
            )
            print("Dataset and text files downloaded and extracted.\n")
        else:
            print("Dataset already exists.")

    def check_and_move_files(self):
        """
        Checks if the required files and directories exist. If not, moves the files and directories
        from the extracted zip directories to the correct locations.
        
        If files are missing, it will move them from the extracted directories to the expected locations.
        """
        print(f"Checking and moving files in {self.path}")

        # Check if required files and directories exist
        required_files = [
            self.path / 'Flicker8k_Dataset',
            self.path / 'Flickr8k_text/Flickr8k.lemma.token.txt',
            self.path / 'Flickr8k_text/Flickr8k.token.txt',
            self.path / 'Flickr8k_text/CrowdFlowerAnnotations.txt',
            self.path / 'Flickr8k_text/Flickr_8k.devImages.txt',
            self.path / 'Flickr8k_text/Flickr_8k.testImages.txt',
            self.path / 'Flickr8k_text/Flickr_8k.trainImages.txt'
        ]
        
        # If all required files and directories exist, do nothing
        if all(file.exists() for file in required_files):
            print("All required files and directories exist. No action needed.")
            return
        
        # Move files if they don't exist
        # Move files from Flickr8k_text.zip directory to one level up
        text_dir = self.path / 'Flickr8k_text.zip'
        for filename in ['Flickr8k.lemma.token.txt', 'Flickr8k.token.txt', 'CrowdFlowerAnnotations.txt', 
                         'Flickr8k.token.txt', 'Flickr_8k.devImages.txt', 'Flickr_8k.testImages.txt', 
                         'Flickr_8k.trainImages.txt']:
            src = text_dir / filename
            dst = self.path / filename
            if src.exists():
                shutil.move(str(src), str(dst))
                print(f"Moved {filename} to {self.path}")
        
        # Check if any directory named 'Flicker8k_Dataset' exists
        dataset_dir = self.path / 'Flicker8k_Dataset'
        if not dataset_dir.exists():
            # If not, go to 'Flickr8k_Dataset.zip' directory and move 'Flicker8k_Dataset'
            zip_dir = self.path / 'Flickr8k_Dataset.zip'
            if zip_dir.exists():
                # Extract the directory from the zip file
                shutil.move(str(zip_dir / 'Flicker8k_Dataset'), str(self.path / 'Flicker8k_Dataset'))
                print("Moved Flicker8k_Dataset from Flickr8k_Dataset.zip directory.")
            else:
                print("Flickr8k_Dataset.zip directory not found.")
        
        return required_files

if __name__ == '__main__':
    dataset_handler = Flickr8kDatasetHandler()
    dataset_handler.get_data()
    # dataset_handler.check_and_move_files()