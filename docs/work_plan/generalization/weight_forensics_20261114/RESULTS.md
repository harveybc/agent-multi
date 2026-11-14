# Saved-Weight Forensics: Measured Results

Full expanded FIT, not the stable-added subset in338. No new account trading return.

|Case|FIT rows|Cost before -> after|Regret before -> after|Better / worse / tied decisions|
|---|---:|---:|---:|---:|
|A_19|196|0.100499732 -> 0.095285494|0.156435285 -> 0.143093188|6 / 2 / 3|
|A_43|196|0.095305457 -> 0.090771205|0.141511388 -> 0.137134156|4 / 1 / 1|
|B_19|392|0.111643395 -> 0.109345482|0.193833237 -> 0.189182553|6 / 2 / 3|
|B_43|392|0.115912840 -> 0.113151363|0.231144661 -> 0.222863006|8 / 4 / 3|

384 finite hybrids,20 saved points,24 transitions. Cross-host continuous maximum difference: 6.94e-12; all common discrete evidence exact.

Complete-arm source-to256 FIT: 43 changed decision occurrences, 9 harmful; 3 harmful changes reduce the same row's surrogate loss. Occurrences are not independent samples.

See [protocol and interpretation](../../339_SAVED_WEIGHT_DECISION_FORENSICS.md).
The16 coalitions per transition are interventions, not independently trained networks. No hybrid is selected or promoted.
