from numpy import around, mean, empty
from transformers import BertTokenizer, BertModel
from newspaper import article as art
import nltk
# import tkinter as tk
nltk.download('punkt')
import torch
import torch.nn as nn

args = {
    'max_len': 200,
    'classes': 2,
    'bert_type': 'bert-base-cased',
    'tokenizer': BertTokenizer.from_pretrained('bert-base-cased')
}

class SentimentClassifier(nn.Module):
    """
    Neural network that classifies news as fake or real using a pretrained
    BERT architecture.
    """
    def __init__(self, args):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(args['bert_type'])
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, args['classes'])
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        _,pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        output = self.dropout(pooled_output)
        output = self.out(output)
        return self.softmax(output)

def get_article_text(url):
    """

    """
    try:
        article = art.Article(url, language="en")
        article.download()
        article.parse()
        article.nlp()
        text = article.text.replace('\n', ' ')
        sentences = nltk.sent_tokenize(text)
        tri_sentences = [" ".join(sentences[i:i+3]) for i in range(len(sentences))]
        return tri_sentences
    except:
        print("Url does not link to a valid article.")
        return None

def encode_text(text):
    encoding = args['tokenizer'].encode_plus(
        text,
        max_length = args['max_len'],
        add_special_tokens=True,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_token_type_ids=False,
        return_tensors='pt',
        truncation=True
    )
    return {
        'input_ids': encoding['input_ids'],
        'attention_mask': encoding['attention_mask']
    }

def is_fake(url):
    """
    <url> is a link to a news article. Returns probability of the
    article being fake news. If <url> is not a link to an article, then returns
    None.
    """
    input = get_article_text(url)
    if input is not None:
        with torch.no_grad():
            predictions = empty(len(input))
            for i, pred in enumerate(input):
                encoding = encode_text(pred)
                input_ids = encoding['input_ids']
                attention_mask = encoding['attention_mask']
                output = model(input_ids=input_ids,attention_mask=attention_mask)
                predictions[i] = output.flatten()[0].item()
                prob = around(mean(predictions)*100, decimals=1)

            label["text"] = "Probability of the article being fake is " + str(prob) + "%"

model = SentimentClassifier(args)
model.load_state_dict(torch.load('model.pt'))
model = model.eval()

# height = 400
# width = 400
# bg = 'green'
#
# root = tk.Tk()
#
# canvas = tk.Canvas(root, height=height, width=width)
# canvas.pack()
#
# frame = tk.Frame(root, bg=bg, bd=5)
# frame.place(relx=0.5, rely=0.1, relwidth=0.75, relheight=0.1, anchor='n')
#
# entry = tk.Entry(frame, font=30)
# entry.place(relwidth=0.65, relheight=1)
#
# button = tk.Button(
#     frame,
#     text="Check Url",
#     font=30,
#     command=lambda: is_fake(entry.get())
# )
# button.place(relx=0.7, relheight=1, relwidth=0.3)
#
# frame_lower = tk.Frame(root, bg=bg, bd=5)
# frame_lower.place(relx=0.5, rely=0.25, relwidth=0.75, relheight=0.1, anchor='n')
#
# label = tk.Label(frame_lower)
# label.place(relwidth=1, relheight=1)
#
# root.mainloop()
