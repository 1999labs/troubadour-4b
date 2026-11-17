# A visit to an unfamiliar settlement

Troubadour practices a narrow version of street smarts: use the encounters you have seen, apply only what they justify, and admit when you cannot tell. This worked example comes from development data, not the reserved final test. It illustrates the task; it is not a model performance result.

Imagine you have arrived in a fictional settlement. A means the locals considered an encounter respectful; B means it breached their etiquette. The model receives a finite menu of possible rule types, but no actual rule for this settlement. These customs are invented and have no connection to a real culture.

| Recipient rank | Setting | Gesture | Prior favor | Recipient origin | Observed judgment |
|---|---|---|---|---|---|
| higher | public | decline | yes | local | A — respect |
| higher | private | decline | yes | visitor | A — respect |
| lower | public | decline | yes | visitor | B — breach |
| higher | private | decline | no | local | A — respect |
| lower | public | accept | no | local | A — respect |
| lower | public | decline | yes | local | A — respect |
| higher | private | accept | no | visitor | B — breach |
| lower | public | accept | yes | visitor | B — breach |

Now consider a **lower-ranked visitor, in public, declining a token, with no prior favor**. That exact encounter did not appear above. What should the model answer?

After checking every allowed rule against the eight observations, exactly two customs remain possible. One says: in public, locals receive A and visitors receive B; in private, declining receives A and accepting receives B. The other says: for a higher-ranked recipient, declining receives A and accepting receives B; for a lower-ranked recipient, locals receive A and visitors receive B.

Both customs fit every observed encounter. Both assign **B** to the new encounter. Therefore B is justified even though the examples have not identified the exact custom. Inferring the answer and identifying the entire rule are different achievements.

Change the query to a **higher-ranked local recipient, in public, accepting a token, with no prior favor**. The first surviving custom assigns A; the second assigns B. The correct response is now **U — insufficient evidence**. Guessing the hidden world's answer would not satisfy the task, because the visible evidence does not distinguish the two possibilities.

## Why more examples matter

| Visible encounters | Customs still consistent | Correct answer to the first query |
|---:|---:|---|
| 0 | 612 | U |
| 1 | 306 | U |
| 2 | 182 | U |
| 4 | 43 | U |
| 8 | 2 | B |

At zero observations, saying U is easy and always correct under this grammar. The interesting question is whether the model starts making justified, correct decisions as relevant evidence arrives. This is why our evaluation separates correct resolutions from unwarranted certainty, rather than relying only on overall accuracy at each observation count.

The checker enumerates the public grammar to establish the answer. The neural predictor never receives its surviving rules, hidden rule, or correct label. It receives only the grammar description, observed encounters, and the new query. The same input and scoring method are used for original Qwen and Troubadour.
