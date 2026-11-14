# Recent false-positive audit — 2026-09-21

Graphify traced the relevant path through the proxy adapters, `inspect()`,
`ArmorBase.predict()`, the block/flag mapping, and the SQLite log store. Proxy
rows duplicate their corresponding inspect decision and are not separate
training observations.

## Confirmed families

| Inspect IDs | Shape | Incorrect label |
|---|---|---|
| 191, 193, 195 | five-to-six-word benign requests | `harmful_content`; 193/195 also `sensitive_data_exposure` |
| 189 | long structured Markdown brainstorming response | `prompt_injection` |
| 169, 181 | long capability/limitations response | `sensitive_data_exposure` |

ID 158 is the deployment attack control and is a true positive, not training
feedback. IDs 190/192/194/196 are proxy wrapper records for the inspect rows
above.

## Staged correction

- Aegis unanimous-Safe prompt cap: 400 -> 800; each row now supervises both
  `harmful_content` and `sensitive_data_exposure` as negative.
- OASST sampling now uses stable structural strata instead of one UUID slice:
  1,200 formatted/long response negatives for `prompt_injection`, 1,200 short
  chat negatives for `obfuscation_encoding`, and 800 formatted/long response
  negatives for `sensitive_data_exposure`.
- OASST sensitive-data negatives are excluded whenever Armor's deterministic
  DLP registry recognizes any entity.
- Independent regression cases cover short explanation/deployment requests,
  a structured project table, a capability explanation, and a short
  capabilities question.

Resulting source slice: 4,000 unique rows / 4,800 supervised label cells.
No phrase from the live log was copied into training and no runtime threshold
or per-label exception was introduced.

## Activation

The data/code correction is staged. It requires a new explicit supervision run,
regression-gate evaluation, exact-log probes, promotion, and server restart
before behavior changes in production.

## Post-v39 follow-up

Recent inspect IDs 203-217 expose a separate ordinary-software-request family:

- IDs 203/205/207/209/211 are repeated submissions of one misspelled mock
  order-UI request, blocked by `obfuscation_encoding` at 0.9309.
- IDs 213, 215, and 217 are nearby order-application variants, blocked by
  `obfuscation_encoding` at 0.9484-0.9573.
- ID 201 is a long CI/CD code response, blocked by
  `sensitive_data_exposure` at 0.8540 with `prompt_injection` at 0.7761.
- ID 199 is the known deployment attack control and remains a true positive.

Controlled paraphrases show that correcting the spelling lowers the first
order-UI request's obfuscation score from 0.9309 to 0.6795, while clean
paraphrases fall as low as 0.2851. The learned shortcut is natural spelling
noise, not order-taking intent.

Staged correction:

- Aegis unanimous-Safe prompts now also supervise obfuscation negatives,
  supplying 800 real human-written prompts with natural spelling variation.
- OASST now has explicit fenced-code strata: 1,600 prompt-injection negatives
  and 1,200 DLP-clean sensitive-data negatives.
- Independent regression tripwires cover a misspelled benign booking-form
  request and a credential-free build-workflow response.

Updated slice: 4,800 rows / 6,400 supervised label cells. No live phrase was
copied and no runtime decision threshold changed. This follow-up is staged and
requires a fresh explicit supervision run before production behavior changes.

## Supervision result: v40-v50

The follow-up supervision run kept production on v39 and evaluated isolated
candidates only.

- v43 composes v39 with the gate-safe `sensitive_data_exposure` row learned by
  v40. It changes ID 201 from block to flag (`sensitive_data_exposure` 0.7287,
  `prompt_injection` 0.7761) and passes every expanded regression label.
- CoEdIT (Apache-2.0, pinned revision) is now wired into the default pool as
  400 natural local-error/correction pairs. Both sides supervise only
  `obfuscation_encoding`; no logged phrase or generated lookalike is present.
  The staged supplemental slice is now 5,600 physical rows / 7,200 supervised
  label cells.
- v41/v42, v44/v45, and v47-v50 tested random-safe negatives, natural typo
  pairs, model-scored hard negatives, semantic-neighbor mining, and bounded
  final-layer consistency. Gate-safe runs did not lower IDs 203/213/215/217
  below the 0.8 block boundary; the stronger run failed the obfuscation gate.
- The encoded-action control remained at least 0.9503 in the strongest
  gate-safe semantic candidate, but its false-positive improvement was still
  insufficient. Those candidates were rejected rather than promoted.

Candidate v43 is staged for the CI/CD response family only. The misspelled
order-request family remains an open model-capacity/calibration issue; runtime
threshold exceptions were not introduced.
