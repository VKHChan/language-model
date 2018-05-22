# language-model
A language model trained on the TED talk transcript

To train the language model,

1. Download the TED talk transcripts (xml file)
2. Run the prepare_data.py
   i. Change the path in the prepare_data.py file to the (xml file)
   ii. Run the prepare_data.py. This cleans the data, trains the tokenizer and prepare the data into sequences of text for training
3. Run the train.py
  This trains an GRU model with 2 hidden layers, 256 hidden neurons, and 0.2 dropout. The trained models will be saved as GRU_weights.{epoch:02d}-{val_loss:.2f}.hdf5}.
4. Generate text by running the generate.py
  i. Change the path name to the trained model.
  ii. Run the file. This will generate 5 samples from the trained model.

#TODO: include all the variables as input arguments to the python code (as oppose to going into the file and changing them everytime
 
