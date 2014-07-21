#!/usr/bin/python
usage="""
./preprocess-corpus.py -v vocab -i infile -o outfile
  Apply the following rules to each token in corpus:
    i) if word is not in the specified vocab file, map it to SURFACE_UNK

    Each output corpus has an accompanying .transforms file containing a list of token transformations
    applied e.g. original word <tab> mapped word
"""
import sys
import optparse
import codecs

SURFACE_UNK = u'<unk>'

def get_string_subject_to_vocab(string, vocab, fallback):
  if string in vocab:
    return string
  else:
    return fallback

def map_tokens(corpus, vocab, out, transform_out):
  write_transform = lambda x,y: transform_out.write(u'%s\t%s\n' % (x,y))
  for line in corpus:
    line = line.strip()
    outline = []
    
    for token in line.split():
      outline.append(get_string_subject_to_vocab(token, vocab, SURFACE_UNK))
      if token != outline[-1]: 
        write_transform(token, outline[-1])

    # write out transformed line
    out.write(u'%s\n' % u' '.join(outline))

if __name__ == '__main__':
  PATHSEP=","
  op = optparse.OptionParser(usage=usage)
  op.add_option("-i", "--input", dest="input", help="Input corpora, string of file names delimited by '" + PATHSEP + "'. The first is used to determine the vocabulary.")
  op.add_option("-o", "--output", dest="output", help="Output files for each corresponding input corpus; list delimited by '" + PATHSEP + "'.")
  op.add_option("-v", "--vocab", dest="vocab", help="Input vocab")
  op.add_option("--surface-unk", dest="surface_unk", help="Symbol to use when mapping away surface forms", default=SURFACE_UNK)
  (options, args) = op.parse_args() 

  if len(args) > 1 and sys.argv[1] == "-h":
    print usage
    sys.exit()

  assert len(options.input.split(PATHSEP)) == len(options.output.split(PATHSEP))
  filecount = len(options.input.split(PATHSEP))
  assert filecount > 0

  SURFACE_UNK = options.surface_unk

  sys.stdout = codecs.getwriter('utf-8')(sys.stdout)

  fileread = lambda f: codecs.open(f, 'r', 'utf-8')
  filewrite = lambda f: codecs.open(f, 'w', 'utf-8')
  first_in_stream = lambda: fileread(options.input.split(PATHSEP)[0])

  vocab = set([x.strip() for x in fileread(options.vocab).readlines()])
  
  for (inf, outf) in zip(options.input.split(PATHSEP), options.output.split(PATHSEP)):
    map_tokens(fileread(inf), vocab, filewrite(outf), filewrite(outf+".transforms"))   


