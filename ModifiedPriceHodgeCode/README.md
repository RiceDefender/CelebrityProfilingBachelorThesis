# CS4100_Project

# Feature extraction code
To run the feature extraction, you need to have a few libraries installed
`pip install spacy`
`pip install emoji`
You'll also need to download the correct spacy model
`python -m spacy download en_core_web_lg`
(this is the model with the necessary word vectors)
Once these models are all installed, run the jupyter notebook cells one by one (ignore the first commented-out cell,
that was to chunk the large file into small, manageable files) When you get to the last cell,
change the directory names in the final process_dirs() call to the directories you actually want to process.
For example, if you're processing dirs '401-600' and '601-800' the call would be
`process_dirs(['401-600', '601-800'])`

This call will output chunked .npy (saved numpy array) files of the generated feature and label vectors for each directory.
The feature array and label array will be in the same order.
They can be loaded into another notebook using np.load('filename') Each label vector is of the form [birthyear, gender_code, occupation_code]

# Model Code
Code for all models can be found in `CelebrityClassification2020Model.ipynb`.

