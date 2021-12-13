import pickle

import numpy as np
import os
import re
import string
import json
from multiprocessing import Pool

import sys
import json
import re
import string


import numpy as np
from numpy.linalg import norm

class DirectedCentralityRnak(object):
    def __init__(self, 
                document_feats,
                extract_num=20,
                beta=0.2, 
                lambda1=1, 
                lambda2=0.8,
                alpha=1,
                processors=8):
        self.extract_num = extract_num
        self.processors = processors
        self.beta = beta
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.alpha = alpha

        self.candidate_phrases = [x['candidate_phrases'] for x in document_feats]
        self.doc_embeddings = [x['sentence_embeddings'] for x in document_feats]
        self.tokens_embeddings = [x['candidate_phrases_embeddings'] for x in document_feats]
    
    def flat_list(self, l):
        return [x for ll in l for x in ll]
    
    def extract_summary(self,):
        paired_scores = self.rank()
        

        rank_list_phrases = []
        for candidate, paired_score in zip(self.candidate_phrases, paired_scores):
            candidates = []
            for i in range(len(candidate)):
                phrase = candidate[i]
                candidates.append([phrase, paired_score[i][0], paired_score[i][1]])
            rank_list_phrases.append(candidates)


        predicted_candidation = []
        for i in range(len(rank_list_phrases)):
            final_score = []
            position_weight = 1 / (np.array(list(range(1, len(rank_list_phrases[i]) + 1))))
            position_weight = np.exp(position_weight) / np.sum(np.exp(position_weight))
            cnt = 0
            for candidate, index, score in rank_list_phrases[i]:
                final_score.append([candidate, score * position_weight[cnt]])
                cnt += 1
            final_score.sort(key = lambda x: x[1], reverse = True)
            candidates = [x[0].strip() for x in final_score]
            predicted_candidation.append(candidates)
        return predicted_candidation


    def pairdown(self, scores, pair_indice, length):
        out_matrix = np.ones((length, length))
        for pair in pair_indice:
            out_matrix[pair[0][0]][pair[0][1]] = scores[pair[1]]
            out_matrix[pair[0][1]][pair[0][0]] = scores[pair[1]]
            
        return out_matrix

    def get_similarity_matrix(self, sentence_embeddings):
        pairs = []
        scores = []
        cnt = 0
        for i in range(len(sentence_embeddings)-1):
            for j in range(i, len(sentence_embeddings)):
                if type(sentence_embeddings[i]) == float or type(sentence_embeddings[i]) == np.float or type(sentence_embeddings[j]) == float or type(sentence_embeddings[j]) == np.float:
                    scores.append(0)
                else:
                    scores.append(np.dot(sentence_embeddings[i], sentence_embeddings[j])) 

                pairs.append(([i, j], cnt))
                cnt += 1
        return self.pairdown(scores, pairs, len(sentence_embeddings))

    def compute_scores(self, similarity_matrix, edge_threshold=0):

        forward_scores = [1e-10 for i in range(len(similarity_matrix))]
        backward_scores = [1e-10 for i in range(len(similarity_matrix))]
        edges = []
        n = len(similarity_matrix)
        alpha = self.alpha
        for i in range(len(similarity_matrix)):
            for j in range(i+1, len(similarity_matrix[i])):
                edge_score = similarity_matrix[i][j]
                # boundary_position_function
                db_i = min(i, alpha * (n-i))
                db_j = min(j, alpha * (n-j))
                if edge_score > edge_threshold:
                    if db_i < db_j:
                        forward_scores[i] += edge_score
                        backward_scores[j] += edge_score
                        edges.append((i,j,edge_score))
                    else:
                        forward_scores[j] += edge_score
                        backward_scores[i] += edge_score
                        edges.append((j,i,edge_score))

        return np.asarray(forward_scores), np.asarray(backward_scores), edges
    
    def _rank_part(self, similarity_matrix, doc_vector, candidate_phrases_embeddings):
        min_score = np.min(similarity_matrix)
        max_score = np.max(similarity_matrix)
        threshold = min_score + self.beta * (max_score - min_score)
        new_matrix = similarity_matrix - threshold
        dist = []
        for emb in candidate_phrases_embeddings:
            if type(doc_vector) == float or type(doc_vector) == np.float or type(emb) == float or type(emb) == np.float:
                dist.append(0)
            else:
                dist.append(1/np.sum(np.abs(emb - doc_vector)))

        forward_score, backward_score, _ = self.compute_scores(new_matrix)

        paired_scores = []
        for node in range(len(forward_score)):
            paired_scores.append([node,  (self.lambda1 * forward_score[node] + self.lambda2 * backward_score[node]) * (dist[node])])

        return paired_scores

    def rank(self,):
        
        similarity_matrix = []
        extracted_list = []
        for embedded in self.tokens_embeddings:
            similarity_matrix.append(self.get_similarity_matrix(embedded))
        for matrix, doc_vector, candidate_phrases_embeddings in zip(similarity_matrix, self.doc_embeddings, self.tokens_embeddings):
            extracted_list.append(self._rank_part(matrix, doc_vector, candidate_phrases_embeddings))
        return extracted_list