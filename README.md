# Translator
English to Malayalam Sentence Translator

## Model Summary

The model takes English Sentences as the input and uses a 256-unit LSTM network to convert the sentence to Malayalam. The input and output are 3D lists with shape = (1, maximum length of sentence, number of words), the optimizer used is adam, and the loss calculated is using categorical cross-entropy. The model uses early stopping and model checkpointing to get the model with the best validation accuracy.

## Steps Involved

1. Extract English and Malayalam sentences from the input files.
2. Create a pandas dataframe with the sentences and convert them to lowercase. Also, prepend "START_ " and append " _END" for the Malayalam sentences.
3. Iterate through the sentences to calculate the maximum length and the number of unique words separately for the 2 languages and create a dictionary to map the words to indexes and vice-versa. 
4. Split the data into training, validation, and test sets and create numpy arrays with the shape = (1, maximum length of sentence, number of words).
5. Write the data into TF Records and save the files along with the dictionaries to a local folder.
6. Read the TF Records and parse the data into the different train, validation, and test datasets.
7. Create a model with 2 LSTM layers having 256 units and set up Model checkpointing and early stopping.
8. Train the model and plot the loss and accuracy using matplotlib.
9. Use the trained model to convert the English sentences in the test dataset and compare them with the expected results to verify the accuracy.

## Future Steps

1. Gather more English and Malayalam sentences for improving accuracy.
2. Clean the data thoroughly to ensure accurate translations.
3. Fine-tune the hyper-parameters.
