# french-gpt
french-gpt is a decoder-only transformer model trained to translate english sentences into french.
The model uses byte-pair encoding to tokenize the dataset before training.
The model was trained using this dataset: https://www.kaggle.com/datasets/dhruvildave/en-fr-translation-dataset

## Usage
Use `tokenizer.py` to split and save the tokens for a dataset
Run `train.py` to train the model on a the tokenized dataset
Run `test.py` to see the model's output for any given input.

## Example
Here is a few sample sentences translated by the model:
```
i need to buy some groceries .
j'ai besoin d'acheter des grosses grosses . 

he is studying hard for his exams .
il étudie dur pour ses examens .   

can you please pass me the salt ?
pouvez -vous me passer le sel ?  

she is learning to play the piano .
elle apprend à jouer du piano .    

we enjoyed a delicious dinner at the restaurant .     
nous avons apprécié un dîner délicieux au restaurant .
```
