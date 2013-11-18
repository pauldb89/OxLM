#!/bin/bash

dataroot="/data/taipan/karher/deep-embeddings/data/alex"
expdir="/data/taipan/karher/deep-embeddings/data/alex/cnlm"

for lang in "de" "en"
do
  for filelist in \
    "train/${lang}.100" "train/${lang}.200" "train/${lang}.500" \
    "train/${lang}.1000" "train/${lang}.5000" "train/${lang}.10000" \
  do
    if [ ${lang} == "de" ]
    then
      appendix="_target"
    else
      appendix="_source"
    fi
    filename="${dataroot}/${filelist}"
    outdir="${expdir}/${filelist}"
    python cldc_processor.py --appendix ${appendix} --datadir "${datadir}/tmp" \
      --postfix "train" ${filename} ${outdir} ${lang}
  done
done

for lang in "de" "en"
do
  filelist="test/${lang}"
  if [ ${lang} == "de" ]
  then
    appendix="_target"
  else
    appendix="_source"
  fi
  filename="${dataroot}/${filelist}"
  outdir="${expdir}/${filelist}"
  python cldc_processor.py --appendix ${appendix} --datadir "${datadir}/tmp" \
    --postfix "test" ${filename} ${outdir} ${lang}
done
