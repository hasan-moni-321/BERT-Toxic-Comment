import transformers
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_LEN = 256
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 10
# BERT_PATH = "bert-base-uncased" # this model is trained only in English language
BERT_PATH = "bert-base-multilingual-uncased"
MODEL_PATH = "/home/hasan/Desktop/tweet_toxic_comment"
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)

