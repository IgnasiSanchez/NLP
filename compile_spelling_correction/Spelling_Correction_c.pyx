from nltk.stem.wordnet import WordNetLemmatizer
import numpy as np
import re
cimport cython
cimport numpy as np


'''
MINIMAL EXAMPLE TO USE THIS CLASS

#import the spelling correction classs
from Spelling_Correction_c  import Spelling_Correction_c 

#Load the words of our corpus
nltk.download('words')
words = nltk.corpus.words.words()

#Create the spelling correction object (it will create the BK tree)
spelling_c = Spelling_Correction_c( words, 1)

#Define a phrase to correct
phrase = "the man wentt to the store to buy a hamer and some nals"

#Correct the phrase
spelling_c.correct_text(phrase)
 
'''


cdef class Spelling_Correction_c:
    '''
    Corrects spelling mistakes by replacing the misspelled words by the closest
    words in the dictionary. It uses a BK-tree structure to store the words and look for the 
    most similar words. The distance between two words is the (weighted) edit distance. 
    
    Args:
        words (list of strings): Words of the vocabulary
        tol (int): Tolerance to calculate similar words with respect to the Edit Distance
        c_ins (int): Inserting cost for Edit Distance
        c_del (int): Deleting cost for Edit Distance
        c_rep (int): Replacing cost for Edit Distance
    
    Attributes:
        words (list of strings): Words of the vocabulary
        tol (int): Tolerance to calculate similar words with respect to the Edit Distance
        c_ins (int): Inserting cost for Edit Distance
        c_del (int): Deleting cost for Edit Distance
        c_rep (int): Replacing cost for Edit Distance
        tree (tuple): BK-tree of the vocabulary 
        
    
    '''
    cdef int c_ins
    cdef int c_del
    cdef int c_rep
    cdef list words
    cdef int tol
    cdef tuple tree
    def __init__(self, list words,int tol,int  c_ins=1,int c_del=1, int c_rep=1):
        self.c_ins = c_ins
        self.c_del = c_del
        self.c_rep = c_rep
        self.words = words
        self.tol = tol

        cdef it = iter(words)
        cdef str root = next(it)
        self.tree = (root, {})
        cdef str i
        
        #Add words to tree
        for i in it:
            self._add_word(self.tree, i)
    
    cdef int editDistance(self,str str1, str str2):
        '''
        Function that calculates the edit distance between two strings. 
        It uses less memory because we only store 2 rows instead of the whole matrix 

        Args:
            str1: (String) First string
            str2: (String) Second string
        
        Returns:
            Edit distance between str1 and str2

        '''
        cdef int m = len(str1)
        cdef int n = len(str2)
        cdef int i,j,del_char, ins_char, rep_char

        # Create a array to memoize distance of previous computations (2 rows) 
        cdef np.ndarray[int,ndim=2] dist = np.zeros([2,m+1], dtype=np.int32)

        # When second string is empty then we remove all characters 
        for i in range(m+1):
            dist[0,i] = i


        # Fill the matrix for every character of the second string
        for i in range(1,n+1):
            #This loop compares the char from second string with first string characters 
            for j in range(m+1):
                #if first string is empty then we have to perform add character peration to get second string   
                if j==0:
                    dist[i%2,j] = i

                #if character from both string is same then we do not perform any operation. We take the diagonal value

                elif str1[j-1] == str2[i-1]:
                    dist[i%2,j] = dist[(i-1)%2,j-1]

                #if the characters are different, we take the minimum of the three edits and addd it with the cost
                else:
                    del_char = dist[(i-1)%2,j] + self.c_del #Delete
                    ins_char = dist[i%2,j-1] + self.c_ins #Insert
                    rep_char = dist[(i-1)%2,j-1] + self.c_rep #Replace

                    dist[i%2,j] = min(del_char, ins_char, rep_char)

        return dist[n%2,m]


    cpdef void _add_word(self,tuple parent, str word):
        '''
        Add word word to tree
        
        Args:
            parent (dict): Parent node from where to start looking
            word (str): word we want to put in the BK-tree
        
        Returns:
            -
        '''
        cdef str pword
        cdef dict children
        cdef int d
        pword, children = parent
        d = self.editDistance(word, pword)        
        
        #If there is not a word at distance d from the parent word, add it
        if d not in children.keys():
            children[d] = (word, {})
            
        #Otherwise, recursively find a word which does not have the same distance to the current word
        #We look at the word which is already at distance d from the parent
        else:
            self._add_word(children[d],word)
            
        
    cpdef list _search_descendants(self, tuple parent, str query_word):
        '''
        Finds words that are similar to query_word (within the tolerance)
        
        Args:
            parent (tuple): Parent node from where to start looking for
            query_word (str): Word from which we want to find similar words
        
        Returns:
            results (list of strings): List of similar words to query_word within the tolerance
        '''
        
        cdef str node_word
        cdef tuple child
        cdef dict children_dict
        cdef int dist_to_node, i 
        cdef list results
        node_word, children_dict = parent
        #Calculate the distance from the query word to the parent word
        dist_to_node = self.editDistance(query_word, node_word)

        results = []
        if dist_to_node <= self.tol: #If the parent word has a distance within the tolerance, we append it to results
            results.append((dist_to_node, node_word))
        
        #We inspect all the words with distance in [d(parent,word) - tol, d(parent,word)+tol]
        for i in range(int(dist_to_node-self.tol), int(dist_to_node+self.tol+1)):
            child = children_dict.get(i) # children_dict[i] can return keyerror
            if child is not None: #For each children within the accepted distances, start again
                results.extend(self._search_descendants(child, query_word))
                
        return results
            
    cpdef list find_closest_neighbours(self, str query_word):
        '''
        Gives ordered list of similar words to query_word
        
        Args:
          query_word (str): Word that we want to compare with others
          
        Returns:
            Ordered list of similar words
        '''
        # sort by distance
        return sorted(self._search_descendants(self.tree, query_word))
    
    cpdef is_number(self, str word):
        cdef res
        try :  
            float(word) 
            res = True
        except : 
            res = False
        return res

    
    cpdef list correct_text(self, list text):
        '''
        Corrects a text by replacing unknown words to their most similar word.
        
        Args:
            text (str): Text to correct
        
        Returns:
            correction (str): Corrected text
        '''
        cdef list correction, w_similar
        cdef str w, w_corrected

        correction = []
        lemmatizer = WordNetLemmatizer()
        #First we find all words (which contain alphanumeric characters or ., and are between spaces)
	
        for w in text:
            if self.is_number(w): #If the word is a number, we leave it that way
                correction.append(w)
            else:
                w_lem = lemmatizer.lemmatize(w.lower(),'v') #Get the root of the word
                if w_lem in self.words: #If the root of the word is in our dict, we leave it that way
                    correction.append(w)
                else:
                    if w.isupper(): #If the word is all in caps, and it has more than one letter
                        correction.append(w) #we assume it is an acronym and leave it that way
                        continue
                    #Otherwise, we look for the most similar word using our BK-tree
                    w_similar = self.find_closest_neighbours(w_lem)

                    if len(w_similar)>0:
                        w_corrected = w_similar[0][1]
                        correction.append(w_corrected)
                    else:
                        # no word found, simply append the unedited word
                        correction.append(w)
        return correction
                    
