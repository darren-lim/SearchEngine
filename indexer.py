import os
import json as js
from bs4 import BeautifulSoup
from bs4.element import Comment
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from collections import OrderedDict
import re
from eventlet.timeout import Timeout


INDEX_SIZE = 10000

class Indexer:

    def __init__(self, exact_list, near_dup_list):
        # Posting structure: {token: {doc index #: [tf, importance of token]
        self.posting_dict = dict()
        self.url_dict = dict()
        self.doc_num = 0  # document numbering or N
        self.fileId = 1

        # https://stackoverflow.com/questions/15547409/how-to-get-rid-of-punctuation-using-nltk-tokenizer
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.stemmer = PorterStemmer()

        self.exact_list = exact_list
        self.near_dups = near_dup_list
        self.list_dup_urls = []

        # for M1 report
        self.doc_count = 0
        self.unique_count = set()


    def indexer_main(self):
        if os.path.exists('DEV'):
            for path in os.listdir('DEV'):
                for json_file in os.listdir('DEV/'+path):
                    # load json
                    json_path = 'DEV/'+path+'/'+json_file
                   # print(json_path)
                    with open(json_path, 'r') as jfile:
                        #with Timeout(5, False):
                        data = js.load(jfile)
                        url = data['url']

                        if url in self.near_dups:
                            continue
                        else:
                            for dup_lst in self.exact_list:
                                if url in dup_lst:
                                    self.near_dups.extend(dup_lst)
                                    self.exact_list.remove(dup_lst)
                                    break

                        self.url_dict[self.doc_num] = url
                        self.parse_html(self.doc_num, data['content'])
                    self.doc_num += 1
                    if self.doc_num % INDEX_SIZE == 0:
                        self.writeIndexToFile()
                        self.fileId += 1
            self.writeIndexToFile()
            print(self.doc_count)
            print(self.doc_num)
            print(len(self.unique_count))
            with open("./FileOutput/urls.txt", "w") as url_txt:
                url_txt.write(str(self.url_dict))
            with open("./FileOutput/report.txt", "w+") as f:
                f.write("number of documents: "+str(self.doc_num)+"\n")
                f.write("number of unique tokens: " + str(len(self.unique_count)))

        else:
            print("Please put DEV folder in the folder with this file.")



    def writeIndexToFile(self):
        print("writing index to file", self.fileId)
        self.posting_dict = OrderedDict(sorted(self.posting_dict.items()))
        #json = js.dumps(self.posting_dict)
        if not os.path.exists("./FileOutput"):
            os.mkdir("./FileOutput")
        f = open("./FileOutput/dict" + str(self.fileId) + ".txt", "w+")
        for key, value in self.posting_dict.items():
            f.write('{"%s":%s}\n' % (key,value))
        #f.write(json)
        f.close()
        self.posting_dict = dict()


    def parse_html(self, doc_index, content):
        soup = BeautifulSoup(content, "lxml")
        #print(doc_index)
        for line in soup.find_all(["h1", "h2", "h3", "strong", "b", "title"]):
            text = line.get_text()
            text = self.removeNonAscii(text)
            token_list = tokenizer(text)
            #token_list = self.tokenizer.tokenize(text)
            for token in token_list:
                self.index_token(token, doc_index, 2)

        all_text = self.find_all_text(soup)
        for words in all_text:
            words = self.removeNonAscii(words)
            tokens = tokenizer(words)
            #tokens = self.tokenizer.tokenize(words)
            for token in tokens:
                self.index_token(token, doc_index, 0)

        #print(self.posting_dict)

        self.doc_count += 1


    def index_token(self, token, doc_index, importance):
        stemmed = self.stemmer.stem(token.lower())
        if not self.check_word(stemmed):
            return
        if stemmed in self.posting_dict:
            if doc_index in self.posting_dict[stemmed]:
                self.posting_dict[stemmed][doc_index][0] += 1
                self.posting_dict[stemmed][doc_index][1] += importance
            else:
                self.posting_dict[stemmed][doc_index] = [1, importance]
        else: # list contains: frequency, weight, tf-idf score
            self.posting_dict[stemmed] = {doc_index: [1, importance]}
        if stemmed not in self.unique_count:
            self.unique_count.add(stemmed)


    '''
    find_all_text has a time complexity of O(N) because we are going through all the text.
    '''
    # https://stackoverflow.com/questions/1936466/beautifulsoup-grab-visible-webpage-text?noredirect=1&lq=1
    def find_all_text(self, soup):
        texts = soup.findAll(text=True)
        visible_texts = filter(self.filter_tags, texts)
        return [t.strip() for t in visible_texts if t.strip() != '']

    '''
    Time complexity of filter_tags is O(1) because in evaluation in set only takes O(1) time.
    '''
    # https://stackoverflow.com/questions/1936466/beautifulsoup-grab-visible-webpage-text?noredirect=1&lq=1
    def filter_tags(self, element):
        if element.parent.name in {'style', 'script', '[document]', 'head', 'title', 'meta', 'noscript', 'h1', 'h2', 'h3', 'strong', 'b'}:
            return False
        if isinstance(element, Comment):
            return False
        return True


    def check_word(self, word):
        if not word.isdigit():
            if len(word) == 1:
                return False
        return True

    # https://stackoverflow.com/questions/1342000/how-to-make-the-python-interpreter-correctly-handle-non-ascii-characters-in-stri
    def removeNonAscii(self, s):
        return "".join(i for i in s if ord(i) < 128)


def makeBookkeeping(finalMerge, bk):


    with open(bk, "w") as bookKeeping:
        read_file = open(finalMerge, "r")
        count = 0

        line = read_file.readline()

        character = (line.split()[0])[2]
        bookKeeping.write(character + " " + str(count) + " ")
        newCharacter = False
        count += len(line)
        while True:

            line = read_file.readline()
            if (line == ""):
                bookKeeping.write(str(count) + " ")
                break

            if (line.split()[0][2] != character):
                newCharacter = True
            else:
                newCharacter = False



            if (newCharacter == True):
                bookKeeping.write(str(count) + " ")
                character = line.split()[0][2]
                bookKeeping.write(line.split()[0][2] + " " + str(count) + " ")
            count += len(line)


    read_file.close()

def merge(file1, file2, writefile):
    with open(writefile, "w") as mergeFile:
        f1 = open(file1, "r")
        f2 = open(file2, "r")
        try:
            merged_list = []
            break_file = 0
            l1 = []
            l2 = []
            while True:
                if len(l1) == 0:
                    line_f1 = f1.readline()
                    if line_f1 == "":
                        print("break line_f1")
                        break_file = 0
                        if len(l2) != 0:
                            mergeFile.write(str(l2[0]) + "\n")
                        break
                    l1.append(eval(line_f1.strip()))
                if len(l2) == 0:
                    line_f2 = f2.readline()
                    if line_f2 == "":
                        print("break line_f2")
                        break_file = 1
                        if len(l1) != 0:
                            mergeFile.write(str(l1[0]) + "\n")
                        break
                    l2.append(eval(line_f2.strip()))

                if list(l1[0].keys())[0] == list(l2[0].keys())[0]:
                    l1[0][next(iter(l1[0]))].update(l2[0][next(iter(l2[0]))])
                    mergeFile.write(str(l1[0]) + "\n")
                    l1.pop(0)
                    l2.pop(0)
                elif list(l1[0].keys())[0] < list(l2[0].keys())[0]:
                    mergeFile.write(str(l1[0]) + "\n")
                    l1.pop(0)
                elif list(l1[0].keys())[0] > list(l2[0].keys())[0]:
                    mergeFile.write(str(l2[0]) + "\n")
                    l2.pop(0)
            if break_file == 0:
                while True:
                    line_f2 = f2.readline()
                    if line_f2 == "":
                        break
                    mergeFile.write(str(eval(line_f2.strip())) + "\n")
            elif break_file == 1:
                while True:
                    line_f1 = f1.readline()
                    if line_f1 == "":
                        break
                    mergeFile.write(str(eval(line_f1.strip())) + "\n")
            f1.close()
            f2.close()
        except EOFError:
            print("end of file")
            f1.close()
            f2.close()

            
def tokenizer(text : "str") -> list:
    data = re.split('[^a-z0-9]+',text.lower())
    data = list(filter(None, data))
    return data


def run_indexer(exact_list, near_dup_list):
    indexer = Indexer(exact_list, near_dup_list)
    indexer.indexer_main()

    merge("./FileOutput/dict1.txt", "./FileOutput/dict2.txt", "./FileOutput/mergedict.txt")
    merge("./FileOutput/mergedict.txt", "./FileOutput/dict3.txt", "./FileOutput/finalmerged.txt")

    makeBookkeeping("./FileOutput/finalmerged.txt", "./FileOutput/bookkeeping.txt")
    print("finished indexer")


if __name__ == "__main__":
    #indexer = Indexer(0,0)
    #indexer.indexer_main()
    makeBookkeeping("./FileOutput/finalmerged.txt", "./FileOutput/bookkeeping.txt")




