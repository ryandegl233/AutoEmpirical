# Paper Dataset Overview

## Overall Counts

| Stage | Records |
| --- | ---: |
| Stage 1 Raw | 35391 |
| Stage 2 Filtered | 4197 |
| Stage 3 Annotated | 2041 |

## Included Papers

| venue   | paper_name                                                                  |   stage1_raw_count |   stage2_filtered_count |   stage3_annotated_count | stage1_to_stage2_filter_rate   | symptom_coverage   | root_cause_coverage   |
|:--------|:----------------------------------------------------------------------------|-------------------:|------------------------:|-------------------------:|:-------------------------------|:-------------------|:----------------------|
| ASE     | Towards understanding the faults of javascript-based deep learning systems  |               4184 |                     683 |                      682 | 83.7%                          | 682/682            | 682/682               |
| ICSE    | IoT Bugs and Development Challenges                                         |               5548 |                     320 |                      320 | 94.2%                          | 320/320            | 320/320               |
| ISSTA   | Bugs in Pods: Understanding Bugs in Container Runtime Systems               |               8275 |                     427 |                      427 | 94.8%                          | 427/427            | 427/427               |
| ICSE    | An Empirical Study on Bugs Inside PyTorch: A Replication Study              |               2207 |                     194 |                      194 | 91.2%                          | 194/194            | 194/194               |
| ICSE    | Understanding Transaction Bugs in Database Systems                          |               7775 |                     140 |                      140 | 98.2%                          | 140/140            | 140/140               |
| FSE     | An Exploratory Study of Autopilot Software Bugs in Unmanned Aerial Vehicles |                567 |                     168 |                      142 | 70.4%                          | 142/142            | 142/142               |
| ICSME   | An Empirical Study on Performance Bugs in Deep Learning Frameworks          |               6835 |                    2265 |                      136 | 66.9%                          | 136/136            | 136/136               |

Stage 2 counts include the 21 records restored by the Stage 2 → Stage 3 lineage repair.
