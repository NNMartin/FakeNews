from numpy import around, mean, empty
from transformers import BertTokenizer, BertModel
from newspaper import article as art
import nltk
import torch
import torch.nn as nn
import tkinter as tk
from typing import List, Dict

nltk.download('punkt')


class SentimentClassifier(nn.Module):
    """
    Neural network that classifies news as fake or real using a pretrained
    BERT architecture.
    """
    def __init__(self, args: dict):
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


class NewsGUI:
    """
    GUI for fake news classification
    """
    ARGS = {
        'max_len': 200,
        'classes': 2,
        'bert_type': 'bert-base-cased',
        'tokenizer': BertTokenizer.from_pretrained('bert-base-cased')
    }
    HEIGHT = 400
    WIDTH = 400
    BG = 'green'

    def __int__(self, model_path='model.pt'):
        """

        :param model_path: relative path to saved PyTorch model.
        :type model_path: str
        """
        self.model = self._init_model(model_path)

    def gui(self):
        """
        GUI for fake news classifier

        :return: None
        """
        def is_fake(url: str):
            """
            Updates the GUI with the probability of the article found at <url>
            being fake news.

            :param url: Url link to a news article.
            :return: None
            """
            input = self.get_article_text(url)
            if input is not None:
                with torch.no_grad():
                    predictions = empty(len(input))
                    for i, pred in enumerate(input):
                        encoding = self.encode_text(pred)
                        input_ids = encoding['input_ids']
                        attention_mask = encoding['attention_mask']
                        output = self.model(
                            input_ids=input_ids, attention_mask=attention_mask
                            )
                        predictions[i] = output.flatten()[0].item()
                        prob = around(mean(predictions) * 100, decimals=1)

                    label["text"] = ("Probability of the article being fake"
                                      f"is {prob}%")

        root = tk.Tk()

        canvas = tk.Canvas(root, height=self.HEIGHT, width=self.WIDTH)
        canvas.pack()

        frame = tk.Frame(root, bg=self.BG, bd=5)
        frame.place(
            relx=0.5, rely=0.1, relwidth=0.75, relheight=0.1, anchor='n'
            )

        entry = tk.Entry(frame, font=30)
        entry.place(relwidth=0.65, relheight=1)

        button = tk.Button(
            frame,
            text="Check Url",
            font=30,
            command=lambda: is_fake(entry.get())
            )
        button.place(relx=0.7, relheight=1, relwidth=0.3)

        frame_lower = tk.Frame(root, bg=self.BG, bd=5)
        frame_lower.place(
            relx=0.5, rely=0.25, relwidth=0.75, relheight=0.1, anchor='n'
            )

        label = tk.Label(frame_lower)
        label.place(relwidth=1, relheight=1)

        root.mainloop()

    @classmethod
    def _init_model(cls, model_path: str) -> nn.Module:
        """
        Initializes neural network

        :param model_path: relative path to saved PyTorch model.
        :return: nn.Module
        """
        model = SentimentClassifier(cls.ARGS)
        model.load_state_dict(torch.load(model_path))
        return model.eval()

    @classmethod
    def encode_text(cls, text: List[str]) -> Dict[str: torch.Tensor]:
        """
        Returns token ids and attention masks of article text

        :param text: Article text as a list where each element is the
                concatenation of 3 sentences.
        :return: Dict[str: torch.Tensor]
        """
        encoding = cls.ARGS['tokenizer'].encode_plus(
            text,
            max_length=cls.ARGS['max_len'],
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

    @staticmethod
    def get_article_text(url: str) -> List[str]:
        """
        Returns list of article text where each element is the
                concatenation of 3 sentences.

        :param url: Url link to a news article.
        :return: List[str] or None
        """
        try:
            article = art.Article(url, language="en")
            article.download()
            article.parse()
            article.nlp()
            text = article.text.replace('\n', ' ')
            sentences = nltk.sent_tokenize(text)
            tri_sentences = [
                " ".join(sentences[i:i + 3]) for i in range(len(sentences))
                ]
            return tri_sentences
        except:
            print("Url does not link to a valid article.")


if __name__ == "__main__":
    news_gui = NewsGUI()
    news_gui.gui()