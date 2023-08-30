from nltk.corpus import stopwords
def clean_text(text_data_df, col):
  for i in range(0,len(text_data_df[col])):
    if type(text_data_df.iloc[i][col]) != float:
      text_data_df.iloc[i][col] = text_data_df.iloc[i][col].lower().replace("\n"," ").replace("\t"," ").strip(" ")
      text_data_df.iloc[i][col] = "".join(c for c in text_data_df.iloc[i][col] if c not in punct)
      filtered_words = [w for w in text_data_df[col].iloc[i].split() if w not in stopwords.words('english')]
      # individual database orinted keywords list is best but due to time constriant could not build it 
      text_data_df.iloc[i][col] = " ".join([c for c in text_data_df[col].iloc[i].split(" ") if not(c[:1].isdigit() and c[1:2] in (p for p in punct))])
      text_data_df.iloc[i][col] = " ".join([w for w in text_data_df[col].iloc[i].split() if w not in stop_words])
      text_data_df.iloc[i][col] = " ".join([w for w in text_data_df[col].iloc[i].split() if w[:-1] not in stop_words])
      text_data_df.iloc[i][col] = " ".join([w for w in text_data_df[col].iloc[i].split() if w not in geo_words])  
      text_data_df.iloc[i][col] = " ".join([w for w in text_data_df[col].iloc[i].split() if w[:1] not in list(map(lambda x: str(x),range(3))) and w[:1] not in list(map(lambda x: str(x),range(4,10)))])  
      text_data_df.iloc[i][col] = " ".join([w for w in text_data_df[col].iloc[i].split(" ") if not(w[:1].isdigit() and w[1:].isalpha())])
      text_data_df.iloc[i][col] = " ".join([w for w in text_data_df[col].iloc[i].split(" ") if not(w[:3].isdigit() and w[3:].isalpha())])
      text_data_df.iloc[i][col] = " ".join([w[:-1] if not(w[:1].isdigit()) and w.endswith(".") else w for w in text_data_df.iloc[i][col].split(" ")])
      text_data_df.iloc[i][col] = " ".join([w for w in text_data_df[col].iloc[i].split(" ") if len(w) > 2 and len(w) < 15])
  return text_data_df
