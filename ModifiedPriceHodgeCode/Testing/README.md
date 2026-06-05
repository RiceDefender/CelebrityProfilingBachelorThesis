To run the code in this package:

1) ```pip install -r requirements.txt```
2) ```python -m spacy download en_core_web_lg```
3) Create a directory that you want the split json data to be stored in
4) ```python data_splitter.py <json_file> <data_directory>```
5) ```python feature_extractor.py <data_directory>```
6) ```python label_prediction.py <initial_file> <output_file>```
