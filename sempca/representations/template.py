import os
import re
from collections import Counter

import numpy as np

from sempca.CONSTANTS import PROJECT_ROOT
from sempca.utils import like_camel_to_tokens, tqdm, get_logger

total_words = 0
num_oov = 0


class TemplateTfIdf:
    def __init__(self):
        self.logger = get_logger("TemplateTfIdf")
        self._word2vec = {}
        self.vocab_size = 0
        self.load_word2vec()

    def not_empty(self, s):
        return s and s.strip()

    def transform(self, words):
        global total_words, num_oov
        if isinstance(words, list):
            return_list = []
            for word in words:
                total_words += 1
                word = word.lower()
                # word = re.sub('[·’!"$%&\'()＃！（）*+,-./:;<=>?，：￥★、…．＞【】［］《》？“”‘\[\\]^_`{|}~]+', '', word)
                if word in self._word2vec.keys():
                    return_list.append(self._word2vec[word])
                else:
                    print(word, end=" ")
                    num_oov += 1
            return np.asarray(return_list, dtype=np.float).sum(axis=0) / len(words)
        else:
            total_words += 1
            word = words.lower()
            # word = re.sub('[·’!"$%&\'()＃！（）*+,-./:;<=>?，：￥★、…．＞【】［］《》？“”‘\[\\]^_`{|}~]+', '', word)
            if word in self._word2vec.keys():
                return self._word2vec[word]
            else:
                num_oov += 1
                return np.zeros(self.vocab_size)

    def load_word2vec(self):
        self.logger.info("Loading word2vec dict.")
        embed_file = os.path.join(PROJECT_ROOT, "datasets/glove.6B.300d.txt")
        if os.path.exists(embed_file):
            with open(embed_file, "r", encoding="utf-8") as reader:
                for line in tqdm(reader.readlines()):
                    tokens = line.strip().split()
                    word = tokens[0]
                    embed = np.asarray(tokens[1:], dtype=np.float)
                    self._word2vec[word] = embed

                    self.vocab_size = len(tokens) - 1

                    if len(tokens) != 301:
                        self.logger.info("Wow: " + line)

        else:
            self.logger.error(
                "No pre-trained embedding file(%s) found. Please check." % embed_file
            )
            exit(2)

    def present(self, id2templates):
        processed_id2templates = {}
        all_tokens = set()
        tokens_template_counter = Counter()

        # Preprocessing templates and calculate token-in-template apperance.
        id2embed = {}
        for id, template in id2templates.items():
            # Preprocess: split by spaces and special characters.
            template_tokens = re.split(r"[,\!:=\[\]\(\)\$\s\.\/\#\|\\ ]", template)
            filtered_tokens = []
            for simplified_token in template_tokens:
                if re.match("[\_]+", simplified_token) is not None:
                    filtered_tokens.append("")
                elif re.match("[\-]+", simplified_token) is not None:
                    filtered_tokens.append("")
                else:
                    filtered_tokens.append(simplified_token)
            template_tokens = list(filter(self.not_empty, filtered_tokens))

            # Update token-in-template counter for idf calculation.
            for token in template_tokens:
                tokens_template_counter[token] += 1
                all_tokens = all_tokens.union(template_tokens)

            # Update new processed templates
            processed_id2templates[id] = " ".join(template_tokens)

        self.logger.info(
            "Found %d tokens in %d log templates"
            % (len(all_tokens), len(processed_id2templates))
        )

        # Calculate IDF score.
        total_templates = len(processed_id2templates)
        token2idf = {}
        for token, count in tokens_template_counter.most_common():
            token2idf[token] = np.log(total_templates / count)

        # Calculate TF score and summarize template embedding.
        for id, template in processed_id2templates.items():
            template_tokens = template.split()
            N = len(template_tokens)
            token_counter = Counter(template_tokens)
            template_emb = np.zeros(self.vocab_size)
            if N == 0:
                id2embed[id] = template_emb
                continue
            for token in template_tokens:
                simple_words = like_camel_to_tokens(token)
                tf = token_counter[token] / N
                if token in token2idf.keys():
                    idf = token2idf[token]
                else:
                    idf = 1
                embed = self.transform(simple_words)
                template_emb += tf * idf * embed
            id2embed[id] = template_emb
        self.logger.info(
            "OOV Rate: %d/%d = %.4f" % (num_oov, total_words, (num_oov / total_words))
        )
        return id2embed
