from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
from string import punctuation
from heapq import nlargest

class FrequencySummarizer:
  def __init__(self, min_cut=0.1, max_cut=0.9):
    """
     Initilize the text summarizer.
     Words that have a frequency term lower than min_cut 
     or higer than max_cut will be ignored.
    """
    self._min_cut = min_cut
    self._max_cut = max_cut 
    self._stopwords = set(stopwords.words('english') + list(punctuation))

  def _compute_frequencies(self, word_sent):
    """ 
      Compute the frequency of each of word.
      Input: 
       word_sent, a list of sentences already tokenized.
      Output: 
       freq, a dictionary where freq[w] is the frequency of w.
    """
    freq = defaultdict(int)
    for s in word_sent:
      for word in s:
        if word not in self._stopwords:
          freq[word] += 1
    # frequencies normalization and fitering
    m = float(max(freq.values()))
    for w in freq.keys():
      freq[w] = freq[w]/m
      if freq[w] >= self._max_cut or freq[w] <= self._min_cut:
        del freq[w]
    return freq

  def summarize(self, text, n):
    """
      Return a list of n sentences 
      which represent the summary of text.
    """
    sents = sent_tokenize(text)
    assert n <= len(sents)
    word_sent = [word_tokenize(s.lower()) for s in sents]
    self._freq = self._compute_frequencies(word_sent)
    ranking = defaultdict(int)
    for i,sent in enumerate(word_sent):
      for w in sent:
        if w in self._freq:
          ranking[i] += self._freq[w]
    sents_idx = self._rank(ranking, n)    
    return [sents[j] for j in sents_idx]

  def _rank(self, ranking, n):
    """ return the first n sentences with highest ranking """
    return nlargest(n, ranking, key=ranking.get)


text = ""
MIT_course = "MIT_lec_1.train"
test_name = "asr-output/eecs183-96.txt"
with open(MIT_course) as f:
    content = f.readlines()
    for line in content:
        text+=line
f.close()
text = "That is, this course is going to be about trade-offs. Given scarce resources, how the individuals and firms trade off different alternatives to make themselves as well-off as possible. That's why economics is called the dismal science. OK? It's called the dismal science because we are not about everyone have everything. We're always the people who say, no, you can't have everything. You have to make a trade- off. OK? You have to give up x to get y. And that's why people don't like us. OK? Because that's why we're called the dismal science, because we're always pointing out the trade-offs that people face.\
\
\
Now, some may call it dismal, but I call it fun. And that may be because of my MIT training, as I said I was an undergraduate here. In fact, MIT is the perfect place to teach microeconomics because this whole institute is about engineering solutions which are really ultimately about constrained optimization. Indeed, what's the best example in the world we have of this? It's the 270 contest. Right? You're given a pile of junk, you've got to build something that does something else. That's an exercise in constrained optimization.\
\
\
All engineering is really constrained optimization. How do you take the resources you're given and do the best job building something. And that's really what microeconomics is. Just like 270 is not a dismal contest, microeconomics is not to me a dismal science. You could think of this course like 270. But instead of the building robots, we're running people's lives. OK? That's, kind of, the way I like to think about this course. Instead of trying to decide how we can build something to move a ping pong ball across a table, we're trying to decide how people make their decisions to consume, and firms make their decisions to produce. That's basically what's going to go on in this class.\
"
fs = FrequencySummarizer()
for s in fs.summarize(text, 2):
   print '*',s
# for article_url in to_summarize[:5]:
#   title, text = get_only_text(article_url)
#   print '----------------------------------'
#   print title
#   for s in fs.summarize(text, 2):
#    print '*',s
