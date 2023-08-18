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
ENG: i need to buy some groceries .
FRE: j'ai besoin d'acheter des grosses grosses . 

ENG: he is studying hard for his exams .
FRE: il étudie dur pour ses examens .   

ENG: can you please pass me the salt ?
FRE: pouvez -vous me passer le sel ?  

ENG: she is learning to play the piano .
FRE: elle apprend à jouer du piano .    

ENG: we enjoyed a delicious dinner at the restaurant .     
FRE: nous avons apprécié un dîner délicieux au restaurant .
```
