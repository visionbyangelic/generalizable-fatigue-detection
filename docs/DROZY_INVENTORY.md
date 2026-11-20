# DROZY dataset: acquisition and inventory

Dataset: DROZY, the ULg Multimodality Drowsiness Database (INTELSIG, University of Liège).
Purpose in this project: second dataset under the locked pre-registration in `docs/dataset2_preregistration.md`.
Date acquired: 20 November 2026.
Status: data acquired and verified. No analysis has been run on this data. This document records only what was downloaded, what the files contain, and which label constructions the data can and cannot support under the locked rules.

## 1. Source and download

The dedicated DROZY server no longer hosts the data. Since July 2018 the database is distributed through ORBi, the open repository of the University of Liège, as stated on the official DROZY site (now at https://www.drozy.uliege.be/).

- ORBi record: https://orbi.uliege.be/handle/2268/191620
- Archive downloaded: https://orbi.uliege.be/bitstream/2268/191620/4/DROZY.zip
- Licence document: https://orbi.uliege.be/bitstream/2268/191620/5/DROZY-LicenseAgreement-20180718.pdf (the same PDF is also bundled inside the archive)
- No form, account, payment, or signature was required. The licence takes effect by downloading (see Section 6).

Local copy (inside the gitignored `data/` tree, never committed):

- `data/raw/drozy/DROZY.zip`: the archive as downloaded, 2,463,610,084 bytes (2.46 GB).
  - MD5: `5f56ff190fa81c3385c33f65662458f7`
  - SHA-256: `0a0bb123f7e3806f6a96af8f14f09b80a75ab41c6b3de7eb90bc207d6e93a98c`
  - Integrity: full CRC test over all 1,769 archive entries passed with no errors.
- `data/raw/drozy/extracted/DROZY/`: the subsets needed for this project, extracted for inspection (77 files): `psg/`, `pvt-rt/`, `KSS.txt`, `README.md`, the licence PDF, and `kinect-intrinsics.yaml`. The rest of the archive (videos, landmark annotations, timestamps, interpolation indices) remains inside the zip only.

Git safety: `git check-ignore` confirms both the archive and the extracted files are covered by the `data/raw/` rule in `.gitignore`, and `git status` shows nothing under `data/` as tracked or untracked. The raw data must never be committed (see the licence terms in Section 6).

## 2. Archive contents

The archive holds one top folder, `DROZY/`, with 1,769 entries. Uncompressed sizes by component:

| Component | Contents | Uncompressed size |
|---|---|---|
| `psg/` | 36 EDF recordings | 244.6 MB |
| `pvt-rt/` | 36 PVT reaction-time CSVs (plus one macOS `._` resource file) | 0.2 MB |
| `KSS.txt` | KSS scores, 14 rows by 3 tests | negligible |
| `videos_i8/` | 36 encoded NIR videos (bundled, not needed for this project) | 1,408.5 MB |
| `annotations-auto/` | frame-level facial landmark annotations | 1,304.3 MB |
| `annotations-manual/` | subject-level landmark annotations with images | 400.4 MB |
| `timestamps/` | per-frame timestamps | 15.3 MB |
| `interpIndices/` | interpolation indices for missing video frames | 10.9 MB |
| `README.md`, licence PDF, `kinect-intrinsics.yaml` | documentation | 0.1 MB |

The raw NIR and depth image folders were already excluded by the provider in the 2018 ORBi release (storage reasons, per the bundled README); only the encoded videos remain.

## 3. Session availability

The design is 14 participants by 3 PVT tests (sessions), 42 in total. The provider documents, in the bundled README, that tests 9-1, 10-2, 12-2, 12-3, and 13-3 were lost in backup incidents, and that test 7-1 never happened. The downloaded data matches that exactly: 36 EDF files and the same 36 PVT CSVs, no more, no fewer.

Sessions present per participant (EDF and PVT CSV agree in every case):

| Participant | Sessions with data | Missing |
|---|---|---|
| 1 | 1, 2, 3 | none |
| 2 | 1, 2, 3 | none |
| 3 | 1, 2, 3 | none |
| 4 | 1, 2, 3 | none |
| 5 | 1, 2, 3 | none |
| 6 | 1, 2, 3 | none |
| 7 | 2, 3 | 1 (never recorded) |
| 8 | 1, 2, 3 | none |
| 9 | 2, 3 | 1 (lost) |
| 10 | 1, 3 | 2 (lost) |
| 11 | 1, 2, 3 | none |
| 12 | 1 | 2, 3 (lost) |
| 13 | 1, 2 | 3 (lost) |
| 14 | 1, 2, 3 | none |

## 4. PSG verification (EDF headers, read with MNE 1.13.2)

All 36 EDF files were opened and their headers read. Findings:

- Sampling rate: 512 Hz in every file, as documented.
- Duration: exactly 600.0 s (10 minutes) in every file.
- Channels in a standard file (11): `Fz`, `Cz`, `C3`, `C4`, `Pz` (EEG, referenced to A1, 10-20 system), `Cam-Sync`, `PVT`, `EOG-V`, `EOG-H`, `EMG`, `ECG`.
- Two files, `1-1.edf` and `2-1.edf`, carry a 12th channel, `Oz` (midline occipital EEG). The bundled README documents this explicitly, so it is expected, not corruption.
- The five EEG channels the pre-registration assumes (Fz, Cz, Pz, C3, C4) are present in every file. Fz, the declared frontal-only channel for this dataset, is present throughout.
- Note for the adapter stage: each EDF also contains a `PVT` channel, a square wave marking stimulus onset and the participant's click, so reaction times exist inside the EDFs as well as in the CSVs.

## 5. Labels

### 5.1 KSS scores (`KSS.txt`)

Format: 14 lines (one per participant), 3 columns (one per test), values 1 to 9. One exception: test 7-1 carries a score of 0, which the provider states is an arbitrary placeholder because that test never happened. It is not a real score and must never be read as one.

Actual values, with the sessions that have signal data:

| Participant | KSS test 1 | KSS test 2 | KSS test 3 | Lowest-KSS session with data | Highest-KSS session with data | Locked contrast constructible? |
|---|---|---|---|---|---|---|
| 1 | 3 | 6 | 7 | test 1 (3) | test 3 (7) | yes |
| 2 | 3 | 7 | 6 | test 1 (3) | test 2 (7) | yes |
| 3 | 2 | 3 | 4 | test 1 (2) | test 3 (4) | yes |
| 4 | 4 | 8 | 9 | test 1 (4) | test 3 (9) | yes |
| 5 | 3 | 7 | 8 | test 1 (3) | test 3 (8) | yes |
| 6 | 2 | 3 | 7 | test 1 (2) | test 3 (7) | yes |
| 7 | 0 (placeholder) | 4 | 9 | test 2 (4) | test 3 (9) | yes, from tests 2 and 3 |
| 8 | 2 | 6 | 8 | test 1 (2) | test 3 (8) | yes |
| 9 | 2 | 6 | 8 | test 2 (6) | test 3 (8) | yes (test 1 lost) |
| 10 | 3 | 6 | 7 | test 1 (3) | test 3 (7) | yes (test 2 lost) |
| 11 | 4 | 7 | 7 | test 1 (4) | tests 2 and 3 tied at 7 | not uniquely: highest is tied |
| 12 | 2 | 5 | 6 | test 1 (2) | none with data | no: only one session has data |
| 13 | 6 | 3 | 7 | test 2 (3) | test 1 (6) | yes (test 3 lost) |
| 14 | 5 | 7 | 8 | test 1 (5) | test 3 (8) | yes |

Every participant's recorded KSS values vary across tests; no participant is flat. The locked rule (within-person lowest-KSS session = alert, highest-KSS session = fatigued, middle session excluded) then runs into two factual problems, flagged here without changing the rule:

1. Participant 11: the highest KSS is tied (tests 2 and 3 both score 7). The locked rule does not specify a tie-break, so the fatigued session is not uniquely defined for this participant.
2. Participant 12: only test 1 has EEG or PVT data. The KSS-highest session (test 3, score 6) has no recording, so no within-person contrast can be formed at all.

Consequence for class prevalence and N: the KSS primary contrast is cleanly constructible for 12 of 14 participants (24 sessions). Participant 11 joins only if a tie-break is decided and recorded as a deviation or clarification; participant 12 cannot join under the locked rule. The pre-registration's feasibility expectation assumed N = 14, so the gate will need recomputing at the realised N, exactly as the pre-registration prescribes. By construction the contrast is balanced within participant: each included participant contributes one alert session and one fatigued session, each 600 s, which is 150 non-overlapping 4 s windows per session before artefact rejection.

### 5.2 PVT reaction times (`pvt-rt/`)

Format: one CSV per session, named `SUBJECT-TEST.csv`, matching the EDF sessions one to one (36 files). The first line is the test start timestamp (`yyyy-mm-dd_hh.mm.ss.mmm`). Each following line holds two timestamps separated by `;`: stimulus time and reaction (mouse click) time. The reaction time is their difference.

Completeness, verified by parsing every file:

- 3,383 stimulus lines in total across the 36 files (86 to 106 per session).
- Every stimulus line has a paired reaction timestamp. There are no missing reactions.
- Reaction times range from 135 ms to 4,494 ms.
- Lapse rates (reaction times above the pre-registered 500 ms threshold) vary widely by session, from 0.0% (session 1-1) to 69.8% (session 4-2), 514 lapses in total.

For the locked RT label arm (sessions ranked within person by lapse rate), two factual cautions:

1. Participant 2 has exactly 2 lapses in each of the three sessions (rates 2.2%, 2.0%, 2.0% on 90, 98, and 99 trials). The ranking is technically defined by rate, but it rests on identical lapse counts separated only by different trial denominators.
2. Participant 13 has exactly 1 lapse in each available session (1.1% of 90 trials in test 1, 0.9% of 106 in test 2). Same caution as above.
3. Participant 12 has only one session of PVT data, so no within-person RT ranking exists either, matching the KSS problem.

All other participants show clearly separated lapse rates across their available sessions.

## 6. Licence and terms

Source: `DROZY-LicenseAgreement-20180718.pdf`, version of 18 July 2018, issued by ULg-INTELSIG and bundled with the data. The agreement takes effect by downloading: "By downloading or using DROZY, the Users agree to the following nine (9) terms". No signature or application was involved.

The terms that bind this project, quoted from the agreement:

- Scope: "The database can be used for research and evaluation purposes only. Specifically, the database cannot be used for the development of commercial products." (Term 4 repeats this as the commercial-use clause.)
- Redistribution: "The Users cannot distribute the database, in full or in part, to any third party without prior written approval from ULg-INTELSIG." (Term 2.) Practical consequence: the raw data stays out of the GitHub repository and off any public link, the same handling as FatigueSet.
- Modification: "The User cannot modify the database in any way." (Term 3.) The downloaded archive is therefore kept byte-identical as the canonical copy; all processing in this project reads from it and writes derived files elsewhere.
- Identity and images: subject identities must not be revealed (Term 5), and database images must not be shown or displayed except those marked publishable (Term 6). This project uses signals and labels only, so both terms are satisfied by design.
- Citation: any document reporting results obtained with DROZY must cite, at minimum (Term 8): Massoz, Q., Langohr, T., François, C., and Verly, J. G., "The ULg Multimodality Drowsiness Database (called DROZY) and Examples of Use", Proceedings of the 2016 IEEE Winter Conference on Applications of Computer Vision (WACV 2016), Lake Placid, NY, March 7-10, 2016.
- Ownership remains with ULg-INTELSIG (Term 1); the database is provided "as is" without warranties (Term 7); ULg may modify or discontinue the database and this agreement (Term 9). The agreement is governed by the laws of Belgium.

## 7. Open items flagged for decision (no rule changed here)

1. Participant 11 KSS tie at the maximum (7 and 7): the locked label rule needs a recorded tie-break decision, or this participant sits out the KSS primary arm.
2. Participant 12 has a single recorded session: unusable for either within-person label arm under the locked rules. Realised N for the KSS arm is 12 clean, 13 at most.
3. Participants 2 and 13 have identical lapse counts across sessions in the RT arm: their session rankings rest on rate denominators only. Worth a recorded note before the RT arm is run.
4. The feasibility gate for DROZY must be recomputed at the realised N before any equivalence language is used, per the pre-registration.
