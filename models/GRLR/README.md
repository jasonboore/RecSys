# recommend_a2c
1.Data preprocessing
	①Run the PMCR/lib/processing.py file to preprocess the raw data
	②Run the PMCR/lib/pmcr.py file to calculate the resources for all items
	③Copy the PMCR/data/all_item_source.csv file to the recommend_a2c/data/net_data folder
	④Run the recommend_a2c/lib/data_splicing.py file to process the graph

2.Model training
	run recommend_a2c/lib/recommend_A2C.py, the recommendations list will be stored in the recommend_a2c/log/ all_recharge.txt in real time
	
	The good models will be saved in the recommend_a2c /model folder during the training.

3.Model testing
	recommend_A2C code is changed, select the way to load the model and comment out the last line of code. Load and run the network model in the Model folder.








