import Stemmer
import re
from collections import defaultdict
import xml.etree.ElementTree as ET
import math


filepath = 'collections/trec.sample.xml'

class SearchEngine:

    def __init__(self, stopwords_path):
        self.stemmer = Stemmer.Stemmer('english')
        self.operators = set(['AND', 'OR', 'NOT'])
        self.stopwords_file = open(stopwords_path ,"r")
        self.stopwords = self.stopwords_file.read()
        self.stopwords_tokens = self.tokenization(self.stopwords)
        self.inverted_index = defaultdict(lambda: defaultdict(list))
        self.all_docs = set()

    def tokenization(self, to_tokenize):
        tokens_matched = re.findall(r'\w+(?:[\'-.,]\w+)*', to_tokenize)
        tokenlist = []
        for i in tokens_matched:
            tokenlist.append(i)
        return tokenlist

    def normalize(self, tokens):
        # Stopping and stemming
        pre_processed = [self.stemmer.stemWord(token).lower()
                            for token in tokens if token.lower() not in self.stopwords_tokens]
        return pre_processed

    def tokenize_and_normalize(self, input_text):
        tokenized = self.tokenization(input_text)
        normalized = self.normalize(tokenized)
        return normalized

    # Parse the XML file
    def parse_xml(self, filepath):
        
        tree = ET.parse(filepath)
        root = tree.getroot()
        doc_nums_set = set()
    # Iterate through each <DOC> element
        for doc in root.findall('DOC'):
            # Extract the DOCNO and Text elements
            docno = doc.find('DOCNO').text

            docno = int(docno)
            
            doc_nums_set.add(docno)
            text = doc.find('TEXT').text
            # text = doc.find('Text').text
            headlines = doc.find('HEADLINE').text
            text = headlines + text

            text = text.strip()

            pre_processed = self.tokenize_and_normalize(text)

            # Generating the positional inverted indexlist
            
            for position, word in enumerate(pre_processed):
                self.inverted_index[word][docno].append(position)
        
        self.all_docs = doc_nums_set
        # return self.inverted_index
        
    def get_doc_nums(self, term_to_find):
        if term_to_find in self.inverted_index:
            return list(self.inverted_index[term_to_find].keys())
        else:
            return []


    # Convert infix to postfix notation
    def infix_to_postfix(self, query):
        output = []
        stack = []
        for token in query.split():
            if token == 'AND' or token == 'OR':
                while stack and stack[-1] in ('AND', 'OR', 'NOT'):
                    output.append(stack.pop())
                stack.append(token)
            elif token == 'NOT':
                stack.append(token)
            else:
                output.append(token)
        while stack:
            output.append(stack.pop())
        return output


    # Evaluate the postfix notation
    def boolean_search(self, postfix_tokens):
        stack = []

        for token in postfix_tokens:
            if token == 'AND':
                right = set(stack.pop())
                left = set(stack.pop())
                stack.append(left.intersection(right))
            elif token == 'OR':
                right = set(stack.pop())
                left = set(stack.pop())
                stack.append(left.union(right))
            elif token == 'NOT':
                term = stack.pop()
                stack.append(self.all_docs.difference(term))
            else:
                stack.append(set(self.inverted_index.get(token, {})))
        return set(stack[0]) if stack else {}

    def phrase_search(self, term1, term2):
        postings1 = self.inverted_index.get(term1, {})
        postings2 = self.inverted_index.get(term2, {})
        
        results = []
        
        for doc in set(postings1.keys()).intersection(postings2.keys()):
            for pos1 in postings1[doc]:
                if (pos1 + 1) in postings2[doc]:
                    results.append(doc)
                    break
        return results

    def proximity_search(self, term1, term2, max_distance=1):
        postings1 = self.inverted_index.get(term1, {})
        postings2 = self.inverted_index.get(term2, {})
        
        results = []
        
        for doc in set(postings1.keys()).intersection(postings2.keys()):
            for pos1 in postings1[doc]:
                for pos2 in postings2[doc]:
                    if abs(pos1 - pos2) <= max_distance:
                        results.append(doc)
                        break
        return set(results)

    def general_search(self, query):
        query = self.query_transformation(query)
        # This function will handle individual search segments and return results.
        def process_segment(segment):
            # Phrase search
            if '"' in segment:
                words = segment.strip('"').split()
                if(len(words) > 1):
                    return set(self.phrase_search(words[0], words[1]))
                else:
                    postfix = self.infix_to_postfix(words[0])
                    return self.boolean_search(postfix)
            # Proximity search
            elif '#' in segment:
                match = re.search(r'#(\d+)\(([^,]+),\s*([^)]+)\)', segment)
                if match:
                    distance, term1, term2 = match.groups()
                    return self.proximity_search(term1.strip(), term2.strip(), int(distance))
            # Boolean/single term search
            else:
                postfix = self.infix_to_postfix(segment)
                return self.boolean_search(postfix)
        
        # Split the query by boolean operators while preserving them.
        segments = re.split(r' (AND|OR|NOT) ', query)

        # Initial result with the first segment.
        results = process_segment(segments[0])

        # Loop through remaining segments and combine results.
        for i in range(1, len(segments), 2):
            operator = segments[i]
            segment_result = set(process_segment(segments[i + 1]))

            if operator == 'AND':
                results &= segment_result
            elif operator == 'OR':
                results |= segment_result
            elif operator == 'NOT':
                print('in NOT')
                results -= segment_result

        return results

    def query_transformation(self, query):
        # Splitting the query into individual tokens, phrases, and proximities
        elements = re.findall(r'\"[^\"]+\"|#\d+\([^)]+\)|\w+(?:[\'-.,]\w+)*', query)
        
        def normalise_term(term):
            if term.lower() not in self.stopwords_tokens:
                return self.stemmer.stemWord(term).lower()
            return ''

        # Process each element
        res = []
        for element in elements:
            if element in self.operators:  # keep boolean operators as-is
                res.append(element)
            elif element.startswith('"'):  # it's a phrase
                phrase_tokens = self.tokenization(element[1:-1])  # excluding quotes
                processed_phrase = ' '.join(
                    [self.stemmer.stemWord(token).lower() for token in phrase_tokens if token.lower() not in self.stopwords_tokens]
                )
                res.append(f'"{processed_phrase}"')
            elif element.startswith('#'):  # it's a proximity
                match = re.search(r'#(\d+)\(([^,]+),\s*([^)]+)\)', element)
                if match:
                    distance, term1, term2 = match.groups()
                    term1 = self.stemmer.stemWord(term1.strip()).lower()
                    term2 = self.stemmer.stemWord(term2.strip()).lower()
                    res.append(f'#{distance}({term1}, {term2})')
            else:  # it's a single term
                res.append(normalise_term(element))

        return ' '.join(res)

    def get_tf_weight(self, term,docnum):
        # print(indexing.get_term_postings(term, index))
        tf_occurences = self.inverted_index.get(term, {}).get(docnum, [])
        tf_weight = (1 + math.log10(len(tf_occurences)))
        return tf_weight

    def get_idf_weight(self, term):
        df = len(self.get_doc_nums(term))
        n = len(self.all_docs)
        idf_weight = math.log10(n/df)
        return idf_weight

    def tfidf_term_weight(self, term, docno):
        tf_weight = self.get_tf_weight(term, docno)
        idf_weight = self.get_idf_weight(term)

        return tf_weight * idf_weight

    def ranked_retrieval(self, query):
        # Pre-process query
        query = self.query_transformation(query)
        query_terms = self.tokenization(query)
        score_dic = {}
        # score = 0

        for i in query_terms:
            # print(i)

            for doc_occurence in self.get_doc_nums(i):
                tfidf_weight = self.tfidf_term_weight(i,doc_occurence)
                current_score = score_dic.get(doc_occurence)

                if current_score:
                    updated_score = current_score + tfidf_weight
                    score_dic[doc_occurence] = updated_score
                else:
                    score_dic[doc_occurence] = tfidf_weight

        ranked_score_list = sorted(score_dic.items(), key= lambda item: item[1], reverse=True)

        return ranked_score_list[:150]

    def write_file(self,filepath_to_write='cw1Results/index.txt'):
        with open (filepath_to_write,'w') as f:
            for term, doc_info in self.inverted_index.items():
                # Write the term and the number of documents it appears in
                f.write(f"{term}:{len(doc_info)}\n")
                
                # Write the document number and positions for each document
                for doc_num, positions in doc_info.items():
                    f.write(f"\t{doc_num}:{','.join(map(str, positions))}\n")

    def get_all_document_numbers(self):
        doc_numbers = set()
        for doc_info in self.inverted_index.values():
            doc_numbers.update(doc_info.keys())
        return set(sorted(doc_numbers))

    def read_boolean_query_file(self, query_filepath, result_filepath):
        with open(query_filepath,'r') as boolean_file:
            queries = boolean_file.readlines()

            with open(result_filepath,'w') as result_file:
                for query in queries:
                    query_split = query.split(maxsplit=1)
                    query_num = query_split[0]
                    query_part = query_split[1].rstrip('\n')
                    boolean_search_result = self.general_search(query_part)
                    for res in boolean_search_result:
                        result_file.write(f'{query_num},{res}\n')

            result_file.close()
        boolean_file.close()

        
    def read_ranked_query_file(self, query_filepath, result_filepath):
        with open(query_filepath,'r') as ranked_ir_file:
            queries = ranked_ir_file.readlines()

            with open(result_filepath,'w') as result_file:
                for query in queries:
                    query_split = query.split(maxsplit=1)
                    query_num = query_split[0]
                    query_part = query_split[1].rstrip('\n')
                    ranked_search_result = self.ranked_retrieval(query_part)
                    for docnum, score in ranked_search_result:
                        result_file.write(f'{query_num},{docnum},{round(score,4)}\n')

            result_file.close()
        ranked_ir_file.close()


    def load_index(self, filepath):
        with open(filepath, 'r') as f:
            current_term = None  # To keep track of the current term while reading the file
            for line in f:
                if not line.startswith('\t'):
                    # New term line
                    current_term, _ = line.strip().split(':')  # We don't need the document count
                else:
                    # Document line
                    doc_info = line.strip().split(':')
                    doc_num = int(doc_info[0])  # Ensure doc_num is an integer
                    positions = list(map(int, doc_info[1].split(',')))  # Ensure positions are integers
                    self.inverted_index[current_term][doc_num] = positions

        self.all_docs = self.get_all_document_numbers()

if __name__ == '__main__':
    # provide the stopwords filepath:
    stopwords_path = "englishST.txt"
    search_engine = SearchEngine(stopwords_path)

    index_choice = int(input('(1) create index or (2) load from saved index amd search? '))
    
    # provide the filepath of the xml that needs to be parsed
    document_filepath = "cw1collection/trec.5000.xml"
    
    # Saved index filepath
    index_filepath = "cw1collection/index.txt"

    # Provide the filepath for the queries
    queries_boolean_filepath = 'cw1collection/queries.boolean.txt'
    # Provide the filepath for the reult to be saved
    results_boolean_filepath = 'cw1collection/results.boolean.txt'

    # Provide the filepath for the queries
    queries_ranked_filepath = 'cw1collection/queries.ranked.txt'
    # Provide the filepath for the reult to be saved
    results_ranked_filepath = 'cw1collection/results.ranked.txt'

    if index_choice == 1:
        search_engine.parse_xml(document_filepath)
        search_engine.write_file(index_filepath)

    elif index_choice == 2:
        search_engine.load_index(index_filepath)
        search_engine.read_boolean_query_file(queries_boolean_filepath,results_boolean_filepath)
        search_engine.read_ranked_query_file(queries_ranked_filepath, results_ranked_filepath)
