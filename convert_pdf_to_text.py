import sklearn
import string, re
import pandas as pd, numpy as np
import PyPDF2
import os, pickle

def convert_pdf_toexcel(dir_path, file_list):
  file_content = []
  for file in files_list:
       content_data= ""
       PDF_fileObj2 = open(dir_path+file, 'rb')
       pdfReader = PyPDF2.PdfFileReader(PDF_fileObj2)
       for i in range(0 , pdfReader.numPages):
           pageObj = pdfReader.getPage(i)
           if i <=3:   #Extracting first 3 pages from PDF
               content_text = pageObj.extractText()
               content_data += content_text
       file_content.append(content_data)
    #Exporting to excel
    pd.DataFrame(file_content).to_excel("train.xlsx")  
