from nltk.stem import PorterStemmer
from bisect import bisect_left
import time
import re
import math
import threading
import queue
lock = threading.Lock()
N = 55393
stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't",
                 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

class search:
    def __init__(self):
        self.stemmer = PorterStemmer()

    def readSearchQuery(self, query):
        query = set(self.tokenizer(query)) - set(stop_words)
        queryList = []
        for q in query:
            if not q.isdigit():
                queryList.append(self.stemmer.stem(q.lower()))
        queryList.sort()
        print(queryList)
        return queryList

    def tokenizer(self, text: "str") -> list:
        text_non_ascii = self.removeNonAscii(text)
        data = re.split('[^a-z0-9]+', text_non_ascii.lower())
        data = list(filter(None, data))
        return data


    # https://stackoverflow.com/questions/1342000/how-to-make-the-python-interpreter-correctly-handle-non-ascii-characters-in-stri
    def removeNonAscii(self, s):
        return "".join(i for i in s if ord(i) < 128)

    def create_bookeeper(self, bookkeeping_path):
        lst = []
        with open(bookkeeping_path, "r") as final_m:
            data = final_m.read()
            # add this to a list of pookkeeping
            lst = data.split()
        return lst

    def load_urls(self, urlPath):
        with open(urlPath, "r") as urlfile:
            url = urlfile.read()
        return eval(url)


    # this function return a list of token_document_dict for all the query words
    # each query word has a vector (tf-idf score for doc1, tfidf score score for doc 2, etc)
    # this vector is token_document_dict {doc1: tf-idf score, doc2: tf-idf score, ...}

    def tf_idf(self, word_posting):
        td_dict_list = []
        query_vector = []

        for q in word_posting:
            token_document_dict = {}
            q = q.split()

            i = 1
            # for z in q:
            #     print(z)

            doc_id_index = 1
            tf_index = 2
            weight_index = 3
            df = (len(q) - 1) // 3

            while doc_id_index < len(q):
                tf = int(q[tf_index])
                weight = int(q[weight_index])

                token_document_dict[q[doc_id_index]] = [(1 + math.log(tf)) * math.log(N / df)+ weight]

                doc_id_index += 3
                tf_index += 3
                weight_index += 3

            td_dict_list.append(token_document_dict)
            query_vector.append(math.log(N / df))

        return td_dict_list, query_vector


    def sort_my_dict(self, my_d, limit):
        i = 0
        results = []
        for k in sorted(my_d, key=my_d.get, reverse=False):
            if (i < limit):
                results.append(k)
            else:
                break
            i += 1
        return results


    def cosine_sim(self, query_vector, doc_vector_dict):
        docID_cosine_dict = {}
        for key, value in doc_vector_dict.items():
            docID_cosine_dict[key] = self.multiply_vectors(query_vector,value)
        return docID_cosine_dict


    def normalize(self, vectorL):
        denom = 0
        results = []
        for ax in vectorL:
            denom += (ax * ax)

        for ax in vectorL:
            results.append(ax / math.sqrt(denom))

        return results



    def makeDocumentVector(self, td_dict_list):
        total_dict = td_dict_list[0]

        for key, value in total_dict.items():
            total_dict[key].extend([0 for _ in range(len(td_dict_list)-1)])

        k = 1

        th_list = []
        while k < len(td_dict_list):
            threads = threading.Thread(target=self.add_vectors_to_td_list, args=(total_dict, td_dict_list[k], k, len(td_dict_list)))
            th_list.append(threads)
            k += 1
        for threads in th_list:
            threads.start()
        for threads in th_list:
            threads.join()

        return total_dict


    def add_vectors_to_td_list(self, t_dict, td_dictlist, ind_k, total_len):
        for key, value in td_dictlist.items():
            if key in t_dict:
                t_dict[key][ind_k] = int(value[0])
            else:
                t_dict[key] = [0 for _ in range(total_len)]
                t_dict[key][ind_k] = int(value[0])



    def multiply_vectors(self, v1, v2):
        if (len(v1) != len(v2)):
            return "cannot multiply. different vectors length"
        results = 0
        for i in range(len(v1)):
            results += v1[i] * v2[i]
        return results


    def BinSearch(self, a, x):
        i = bisect_left(a, x)
        if i != len(a) and a[i] == x:
            return i
        else:
            return -1


    def findURL(self, docIDResults, URLFile):
        URL = []

        for id in docIDResults:
            URL.append(URLFile[id])
        return URL


    def add_dups(self, docIDList):
        merged_list = set.intersection(*docIDList)
        return merged_list


    def intersect_sets(self, set1, set2):
        if len(set1) == 0:
            return set2
        return set1.intersection(set2)


    def calculate_tfidf_cosine(self, word_posting_list):
        tf_idf_thing = self.tf_idf(word_posting_list)
        td_dict_list = tf_idf_thing[0]
        query_vector = self.normalize(tf_idf_thing[1])
        d_vector_dict = self.makeDocumentVector(td_dict_list)
        for id, doc_vector in d_vector_dict.items():
            doc_vector = self.normalize(doc_vector)
            d_vector_dict[id] = doc_vector
        cosine_vector = self.cosine_sim(query_vector, d_vector_dict)
        return cosine_vector


    def final_search_file(self, bookkeeping, finalMerge, word, queue) -> tuple:

        #lock.acquire()
        with open(finalMerge, "r") as final_marg:
            # m is the word
            first_char = word[0]
            start = 0
            end = 0
            if first_char in bookkeeping:
                # print(first_char)
                ind = bookkeeping.index(first_char)
                # print(ind)
                start = int(bookkeeping[ind + 1])
                end = int(bookkeeping[ind + 2])
            final_marg.seek(start)
            count = start
            sizeWord = len(word)

            while count < end:
                line = final_marg.readline()
                if line == "":
                    break

                if line[:sizeWord] == word:
                    word_line = line.split()
                    ind = 1
                    doc_set = set()
                    while(ind < len(word_line)):
                        doc_set.add(int(word_line[ind]))
                        ind+=3

                    queue.put((line.strip(),doc_set))

                    #print("done with final search", time.time() - startTime, "\n")
                    return (line.strip(), doc_set)

                count += len(line)


        print("cannot find word ", word, "\n")
        #lock.release()
        return "", set()


if __name__ == "__main__":
    print("Enter nothing to quit")
    searcher = search()
    book = searcher.create_bookeeper("./FileOutput/bookkeeping.txt")
    url_file = searcher.load_urls("./FileOutput/urls.txt")
    N = len(url_file)
    print(N)

    while(True):
        query = input("search query : ")
        if query == "":
            break
        qList = searcher.readSearchQuery(query)
        starttime = time.time()
        if len(qList) == 0:
            print("No valid query tokens. Could not find any results. Try again.")
            continue

        que = queue.Queue()
        thread_list = []
        for word in qList:
            thread = threading.Thread(target=searcher.final_search_file, args=(book, "./FileOutput/finalmerged.txt",word,que))
            thread_list.append(thread)
        for thread in thread_list:
            thread.start()
        for thread in thread_list:
            thread.join()

        thread_list.clear()

        query_word_posting = []
        query_doc_setlist = []
        while not que.empty():
            posting = que.get()
            query_word_posting.append(posting[0])
            query_doc_setlist.append(posting[1])

        if len(query_word_posting) == 0:
            print("No valid tokens, please try again")
            continue

        merged_doc_set = searcher.add_dups(query_doc_setlist)
        cosine_vector = searcher.calculate_tfidf_cosine(query_word_posting)
        cosine_vector_keys = set(int(i) for i in cosine_vector.keys())
        final_doc_set = merged_doc_set.intersection(cosine_vector_keys)
        new_cos_vector = dict()

        for id in final_doc_set:
            new_cos_vector[id] = cosine_vector[str(id)]

        d = searcher.sort_my_dict(new_cos_vector, 5)
        URL = searcher.findURL(d, url_file)
        endtime = time.time() - starttime
        print("------------\nTotal Time Elapsed = ", endtime)

        i = 0
        print("\n************  SEARCH RESULTS   ************* \n\n")
        for u in URL:
            if (i < 5):
                print(u, "\n")
            i += 1
