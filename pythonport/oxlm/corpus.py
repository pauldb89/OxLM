class Dict(object):
    """Dict stores/constructs a vocabulary list, and preserves a mapping from words to word ids."""
    def __init__(self, b0_="<bad0>", sos_="<s>", eos_="</s>", bad0_id_=-1):
        if type(b0_) is not str:
            raise TypeError("Expecting b0_ to be of type str. Type was: %s." % type(b0_).__name__)
        if type(sos_) is not str:
            raise TypeError("Expecting sos_ to be of type str. Type was: %s." % type(sos_).__name__)
        if type(eos_) is not str:
            raise TypeError("Expecting eos_ to be of type str. Type was: %s." % type(eos_).__name__)
        if type(bad0_id_) is not int:
            raise TypeError("Expecting bad0_id_ to be of type str. Type was: %s." % type(bad0_id_).__name__)

        super(Dict, self).__init__()
        self.__b0_ = b0_
        self.__sos_ = sos_
        self.__eos_ = eos_
        self.__bad0_id_ = bad0_id_

        self.__words_ = []
        self.__d_ = {}


    def min(self):
        return 0


    def max(self):
            return len(self.__words_) - 1


    def size(self):
        return len(self.__words_)


    def ConvertWhitespaceDelimitedLine(self, line):
        """Maps a line to a list of word ids."""
        if type(line) is not str:
            raise TypeError("Expecting line to be of type str. Type was: %s." % type(line).__name__)

        parts = line.strip().split() # tokenise
        out = map(self.Convert, parts) # out->clear(); ++ assignment to out vector
        return out


    def Lookup(self, word):
        """Looks up the id of a word, and returns bad id if not found."""
        if type(word) is not str:
            raise TypeError("Expecting input to be of type str. Type was: %s." % type(word).__name__)
        return self.__d_.get(word, self.__bad0_id_) # could be inlined


    def Convert(self, input, frozen=False):
        """Converts input word to word id (adding it to Dict) if frozen==False, or input word id to word."""
        if type(input) is str:
            word = input
            if word not in self.__d_:
                if frozen:
                    return self.__bad0_id_
                else:
                    self.__words_.append(word)
                    self.__d_[word] = len(self.__words_) - 1
                    return len(self.__words_) - 1
            else:
                return self.__d_[word]

        elif type(input) is int:
            id = input
            if not self.valid(id):
                return self.__b0_
            else:
                return self.__words_[id]
        else:
            raise TypeError("Expecting input to be of type str or int. Type was: %s." % type(input).__name__)


    def getVocab(self):
        """Returns the vocabulary in the Dict."""
        return self.__words_ # could be inlined


    def valid(self, id):
        """Checks if a given id is valid."""
        return id >= 0 # could be inlined

def ReadFromFile(filename, d, src, src_vocab):
    """Converts a text file to a list of arrays of word ids. Returns the list of arrays."""
    if type(filename) is not str:
        raise TypeError("Expecting filename to be of type str. Type was: %s." % type(filename).__name__)

    if type(d) is not Dict:
        raise TypeError("Expecting d to be of type Dict. Type was: %s." % type(d).__name__)

    if type(src) is not list:
        raise TypeError("Expecting src to be of type str. Type was: %s." % type(src).__name__)

    if type(src_vocab) is not set:
        raise TypeError("Expecting src_vocab to be of type list. Type was: %s." % type(src_vocab).__name__)

    if len(src) > 0:
        while len(src) > 0: src.pop() # src->clear();
    with open(filename, 'r') as f:
        for line in f:
            converted_line = d.ConvertWhitespaceDelimitedLine(line)
            src.append(converted_line)
            src_vocab.update(set(converted_line))
    return (src, src_vocab)
