# Paper Dataset Overview

## Selection Rule

The final dataset excludes papers whose Stage 1 to Stage 2 filtering rate is 0%. Each retained paper has raw data, a human-filtered set, a human-annotated set, and explicit symptom/root_cause labels in the final records.

## Overall Counts

| Stage             | Records |
| ----------------- | ------- |
| Stage 1 Raw       | 33822   |
| Stage 2 Filtered  | 4199    |
| Stage 3 Annotated | 2050    |

Final annotated dataset file: `Dataset/stage3.csv`

## Included Papers

| Venue | Raw Data Time Range      | Paper                                                                       | Stage 1 Raw | Stage 2 Filtered | Stage 3 Annotated | S1->S2 Filter Rate | Symptom Coverage | Root Cause Coverage |
| ----- | ------------------------ | --------------------------------------------------------------------------- | ----------- | ---------------- | ----------------- | ------------------ | ---------------- | ------------------- |
| ASE   | 2018-03-27 to 2021-12-23 | Towards understanding the faults of javascript-based deep learning systems  | 3859        | 684              | 682               | 82.3%              | 682/682          | 682/682             |
| ICSE  | 2012-07-27 to 2020-03-13 | IoT Bugs and Development Challenges                                         | 5565        | 323              | 320               | 94.2%              | 320/320          | 320/320             |
| ISSTA | 2021-06-01 to 2023-05-31 | Bugs in Pods: Understanding Bugs in Container Runtime Systems               | 8271        | 429              | 429               | 94.8%              | 429/429          | 429/429             |
| ICSE  | up to 2022-10-20         | An Empirical Study on Bugs Inside PyTorch: A Replication Study              | 2205        | 194              | 194               | 91.2%              | 194/194          | 194/194             |
| ICSE  | 2018-01 to 2022-12       | Understanding Transaction Bugs in Database Systems                          | 7775        | 140              | 140               | 98.2%              | 140/140          | 140/140             |
| FSE   | not published in source  | An Exploratory Study of Autopilot Software Bugs in Unmanned Aerial Vehicles | 569         | 168              | 142               | 70.5%              | 142/142          | 142/142             |
| ICSME | 2016-08-16 to 2021-03-16 | An Empirical Study on Performance Bugs in Deep Learning Frameworks          | 5578        | 2261             | 143               | 59.5%              | 143/143          | 143/143             |

## Excluded Papers

| Paper ID                                          | Reason                                          |
| ------------------------------------------------- | ----------------------------------------------- |
| fse2023_understanding_the_bug_characteristics_and | S1=395 and S2=395, so S1->S2 filter rate is 0%. |
| icse2022_characterizing_and_detecting_bugs_in     | S1=83 and S2=83, so S1->S2 filter rate is 0%.   |

## Selection Notes

| Paper ID                                            | Selection Note                                                                                                                                                                  |
| --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ase2022_towards_understanding_the_faults_of         | Retained from previous final dataset; S1>S2 and symptom/root_cause labels are present.                                                                                          |
| icse2021_iot_bugs_and_development_challenges        | Retained from previous final dataset; S1>S2 and symptom/root_cause labels are present.                                                                                          |
| issta2024_bugs_in_pods_understanding_bugs           | Retained from previous final dataset; S1>S2 and symptom/root_cause labels are present.                                                                                          |
| icse2023_an_empirical_study_on_bugs                 | Retained from previous final dataset; S1>S2 and symptom/root_cause labels are present.                                                                                          |
| icse2024_understanding_transaction_bugs_in_database | Retained from previous final dataset; S1>S2 and symptom/root_cause labels are present.                                                                                          |
| fse2021_an_exploratory_study_of_autopilot           | Added replacement dataset; raw bug set has 569 issues, final taxonomy has 168 records, and 142 records have explicit symptom/root_cause labels after removing unclear symptoms. |
| icse2022_an_empirical_study_on_performance          | Added replacement dataset; performance bug information files form the human-filtered set and perf_bugs_taxonomy.csv provides annotated root causes.                             |
