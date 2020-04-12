# DS-DisasterResponse

This contains the Disaster Reponse Pipeline projects files  

__File strucutures:__  

data  
|-- categories.csv (training data labels)  
|-- messages.csv (training data)   
|-- DisasterResponse.db (database storing cleaned data)    
|-- process_data.py (codes to clean and save the training data)  

models  
|-- message_model (final model for classification)  
|-- train_classifier.py (codes to build and save the model)  

app  
|-- templates  
|-- |-- go.html (classification results page)  
|-- |-- master.html (starting page)  
|-- run.py (running the app)  

prep codes  
|-- ETL Pipeline Preparation.ipynb (draft codes for data processing)  
|-- ML Pipeline Preparation.ipynb (draft codes for model training and tuning)  

__Notes:__  
The final Ridge model takes forever to run but perfroms better than the tuned random forest

