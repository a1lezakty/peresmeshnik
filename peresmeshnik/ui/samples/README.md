# Test phrases for the setup wizard

One FLEURS sentence (dev split, id 1515: "It is Martelly's fifth CEP in four years.") in each of the 14 languages, 16 kHz
mono Ogg Opus:

- `en.ogg`, `ru.ogg`: cut from `tests/data/call-*.ogg` ("Это пятый CEP Мартейи за четыре года.");
- the others: the first recording of sentence 1515 in FLEURS `uk_ua`, `de_de`, `fr_fr`, `es_419`, `it_it`, `pt_br`,
  `pl_pl`, `tr_tr`, `cmn_hans_cn` (`zh.ogg`), `ja_jp`, `ko_kr`, `ar_eg`, with the silence before and after it trimmed.

FLEURS: Conneau et al., 2022, https://huggingface.co/datasets/google/fleurs, licensed CC-BY-4.0.
