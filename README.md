# IDF_document_classifier_assigment
IDFY assignment for document classification NLP domain

Here I have used a very simple and fast technique to classify whether a form or not is cosine similarity.
Steps to use the code :

1) Convert pdf to excel file using convert_pdf_to_text.py using convert_pdf_toexcel(dir_path, file_list) function returns excel file
2) Clean the extracted txt using clean_text.py using clean_text(text_data_df, col)
3) Classify using cosine similarity metrics classifier.py using get_form_or_not(df, col) , it returns df with filename, score and class output.0.7 is used as threshold as it gives best accuracy 

Improvements :

1) Instead of countvectorizer which is a content based extractor could have used tfid as context-based extractor
2) Instead  of simple formulae based classifier transformer based model which uses a positional encoder is best suitable but it will require lots of labelled data
3) Even a one shot leaner based approach is best like siamese model for getting the fast and accurate output .
