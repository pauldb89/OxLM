
#Usage: ./preprocess-corpus-count.sh  CORPUS MIN_FREQ

tr ' ' '\n' < $1 | awk '{x[$1]++} END {for (w in x){ print x[w] " " w}}' > counts
perl -ne 'm/^(\d+) (.*)/; if ($1 >= '$2') {print "$2\n"};' < counts | sort > vocab
