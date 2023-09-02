# IDF_document_classifier_assigment
IDFY assignment for document classification NLP domain

Here I have used a very simple and fast technique to classify whether a form or not is cosine similarity.
Steps to use the code :

1) Convert pdf to excel file using convert_pdf_to_text.py using convert_pdf_toexcel(dir_path, file_list) function returns excel file
2) Clean the extracted txt using clean_text.py using clean_text(text_data_df, col)
3) Classify using cosine similarity metrics classifier.py using get_form_or_not(df, col), it returns df with filename, score, and class output.0.7 is used as the threshold as it gives the best accuracy
4) It compares the score with six form types finds the max score and assigns the class, if the score is less than 0.7 then it assigns others as class

Improvements :

1) Instead of a count vectorizer which is a content-based extractor could have used tfid as a context-based extractor
2) Instead  of a simple formulae-based classifier transformer model which uses a positional encoder is best suitable but it will require lots of labeled data
3) Even a one-shot leaner-based approach is best like a Siamese model for getting a fast and accurate output.
4) Even instead of comparing with all the documents for cosine metrics we can first form clusters and compare the document in the respective cluster where our reference file exists 
