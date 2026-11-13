# Thesis prompt data

`thesis_gender_bias_200.json` is copied without content changes from the previous public project at [Mechanistic-Intervention-Bias/data/gender_bias.json](https://github.com/aniruddhak94/Mechanistic-Intervention-Bias/blob/main/data/gender_bias.json). It contains 200 occupation prompts paired with neutralized controls and the prior expected-direction annotations.

The Kaggle notebook splits by occupation so repeated wording variants never cross train, validation, and test. It reports both the prior first-token L2 metric and a separate tokenization-aware pronoun-continuation metric. The test portion is only about 40 prompts/20 occupation groups, so it is a careful replication extension, not broad demographic evidence. The separate 15-example demographic set is too small for confirmatory cross-model conclusions and is not included in the default study.

The matched clean/control labels are useful for representation fitting and a narrow gate diagnostic. They do not constitute an independently annotated, general-purpose bias-context dataset.
