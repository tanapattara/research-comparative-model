# รายงานสรุปผลการทดลอง E2–E5

**โครงการ:** Multi-Station, Multi-Horizon Water-Level Forecasting at Nong Khai  
**ขอบเขตข้อมูล:** expanding test folds A–D และ final-period test ปี 2025  
**สถานะ:** E2–E5 รันครบในขอบเขต fixed-configuration โดยมี protocol deviation ที่ระบุไว้ในหัวข้อ 8

## 1. ลำดับการตัดสินใจ

การทดลองรักษาลำดับ validation ก่อน test ดังนี้:

1. E2 เลือก architecture จาก validation โดยยังไม่เปิด test
2. E3 เลือก look-back จาก validation ด้วย one-standard-error rule
3. E4 ทำ station ablation จาก validation โดยตรึง architecture, look-back และ hyperparameters
4. E5 เปิด test folds หลัง E4 เสร็จแล้ว และรันทุก seed ที่ประกาศไว้โดยไม่เลือกเฉพาะ seed ที่ดีที่สุด

configuration สุดท้ายคือ **MIMO + LSTM + look-back 7 วัน** ใช้ seeds `42, 52, 62, 72, 82`

## 2. E2 — การเลือก architecture

ค่า objective คือ MAE เฉลี่ยของวัน 7 และ 14 บน validation folds A–D:

| Architecture | Objective MAE (m) |
|---|---:|
| LSTM | 0.775664 |
| Compact Transformer | 0.815374 |
| TCN-GRU | 0.973882 |

จึงเลือก **LSTM** สำหรับ phase ถัดไป ผลนี้เป็นการเปรียบเทียบ frozen compact configurations ไม่ใช่ผลจาก Optuna 30 trials

## 3. E3 — การเลือก look-back

| Look-back (วัน) | Objective MAE (m) | Fold SE (m) | อยู่ภายใน 1 SE ของค่าดีที่สุด |
|---:|---:|---:|---|
| 7 | 0.806280 | 0.025319 | ใช่ |
| 14 | 0.824388 | 0.027518 | ไม่ใช่ |
| 30 | 0.820549 | 0.054428 | ไม่ใช่ |
| 60 | 0.775664 | 0.039210 | ใช่ |

ค่าดีที่สุดโดยตรงคือ 60 วัน แต่ threshold ของ one-SE rule เท่ากับ 0.814875 m และ 7 วันอยู่ภายใน threshold จึงเลือก **ช่วงสั้นที่สุด 7 วัน** ตามกฎที่กำหนดไว้ล่วงหน้า

## 4. E4 — ผลของสถานีต้นน้ำ

ค่าบวกหมายถึงสถานีช่วยลด MAE; ค่าลบหมายถึง MAE แย่ลงใน validation

### Incremental addition

| สถานีที่เพิ่ม | Day 7 (m) | Day 14 (m) |
|---|---:|---:|
| VIE | -0.018893 | 0.005481 |
| CKH | 0.057700 | 0.029302 |
| LUA | 0.032099 | 0.022130 |
| CSA | 0.081753 | 0.011337 |

### Leave-one-out

| สถานีที่ตัดออก | Day 7 (m) | Day 14 (m) |
|---|---:|---:|
| CSA | 0.081753 | 0.011337 |
| LUA | 0.014846 | 0.053518 |
| CKH | 0.013812 | 0.005695 |
| VIE | -0.005995 | -0.009548 |

CKH ให้ประโยชน์ค่อนข้างสม่ำเสมอใน incremental analysis ส่วน CSA เด่นที่ day 7 และ LUA เด่นที่ day 14 การตัด VIE ออกไม่ได้ทำให้ผลแย่ลงในสอง horizon หลัก อย่างไรก็ตาม E4 ใช้ development seed 42 จึงควรตีความเป็น station screening ไม่ใช่หลักฐานยืนยันสุดท้าย

## 5. E5 — ผลประเมินสุดท้าย

ตารางต่อไปนี้เป็น mean ± SD ของ MAE ระหว่าง 5 folds × 5 seeds:

| Horizon | S0: NON only (m) | S4: ทุกสถานี (m) |
|---:|---:|---:|
| 1 | 0.177390 ± 0.065062 | 0.391661 ± 0.188604 |
| 3 | 0.400824 ± 0.146735 | 0.447585 ± 0.195024 |
| 5 | 0.570948 ± 0.212791 | 0.529232 ± 0.228958 |
| 7 | 0.688541 ± 0.234267 | 0.635686 ± 0.258134 |
| 14 | 0.949085 ± 0.189763 | 0.923183 ± 0.249613 |

S4 แย่กว่า S0 ที่ horizon สั้น แต่เริ่มให้ MAE ต่ำกว่าที่วัน 5, 7 และ 14

### Paired S4 versus S0 inference

ผลด้านล่างจับคู่ issue date เดียวกัน หลังเฉลี่ย prediction ระหว่าง seeds และใช้ moving-block bootstrap 2,000 ครั้ง:

| Horizon | MAE improvement ของ S4 | Relative improvement | 95% CI, block 30 วัน | Holm p-value | ข้อสรุป |
|---:|---:|---:|---:|---:|---|
| 7 | 0.034958 m | 5.35% | [-0.019899, 0.079278] | 0.254930 | ถึงเกณฑ์ 5% แต่ CI คร่อมศูนย์ |
| 14 | 0.017764 m | 1.91% | [-0.041699, 0.055208] | 0.473705 | ไม่ถึง 5% และ CI คร่อมศูนย์ |

ดังนั้น **ยังไม่มีหลักฐานเชิงสถิติที่ยืนยันว่า S4 ดีกว่า S0** ที่วัน 7 หรือ 14 ตาม decision rule แม้ค่าเฉลี่ยจะเป็นประโยชน์เล็กน้อย

### เปรียบเทียบกับ Ridge

Ridge-S4 มี MAE เฉลี่ย 0.619141 m ที่ day 7 และ 0.904699 m ที่ day 14 ขณะที่ LSTM-S4 ได้ 0.635686 และ 0.923183 m ตามลำดับ

paired comparison แสดงว่า LSTM แย่กว่า Ridge โดยเฉลี่ย 0.012107 m ที่ day 7 และ 0.021118 m ที่ day 14 แต่ confidence intervals ยังคร่อมศูนย์ทั้งคู่ ดังนั้น **ไม่มีหลักฐานว่า LSTM เหนือกว่า Ridge และไม่มีหลักฐานชัดว่าต่างกัน**

### Predictively useful horizon

เมื่อกำหนดให้ horizon มีประโยชน์เมื่อ 95% CI ของ MAE skill เทียบ persistence สูงกว่าศูนย์และ NSE สูงกว่าศูนย์:

| Horizon | MAE skill | 95% CI | NSE | ผ่านเกณฑ์ |
|---:|---:|---:|---:|---|
| 5 | 8.18% | [-3.97%, 18.05%] | 0.929978 | ไม่ผ่าน |
| 6 | 9.74% | [0.46%, 18.04%] | 0.914483 | ผ่าน |
| 7 | 8.93% | [-0.24%, 16.75%] | 0.897900 | ไม่ผ่าน |
| 14 | 4.76% | [-2.45%, 11.85%] | 0.790792 | ไม่ผ่าน |

farthest horizon ที่ผ่านกฎคือ **6 วัน** จึงเรียกว่า predictively useful horizon ได้ 6 วัน ไม่ควรเรียกว่า operationally useful จนกว่าผู้ใช้ระบบเตือนภัยจะยอมรับ threshold และประเมินการใช้งานจริง

### Final period ปี 2025

เฉพาะ final-period 2025 ค่า MAE ของ S4 เท่ากับ 0.218668, 0.227472, 0.245319, 0.293840 และ 0.578312 m ที่วัน 1, 3, 5, 7 และ 14 ตามลำดับ S4 แย่กว่า S0 ที่วัน 1–3 แต่ดีกว่าที่วัน 5–14

ปี 2025 ต้องเรียกว่า **final-period test** ไม่ใช่ untouched holdout เพราะ Paper 1 เคยตรวจช่วงนี้แล้ว

## 6. High-water และ event evaluation

ผลเฉลี่ยที่ training-only 95th-percentile threshold:

| Station set | Horizon | POD/Recall | FAR | CSI | F1 |
|---|---:|---:|---:|---:|---:|
| S0 | 7 | 0.3828 | 0.3783 | 0.3037 | 0.4136 |
| S4 | 7 | 0.5971 | 0.3786 | 0.4365 | 0.5937 |
| S0 | 14 | 0.1822 | 0.5214 | 0.1374 | 0.2228 |
| S4 | 14 | 0.1721 | 0.4839 | 0.1380 | 0.2262 |

S4 ช่วย event detection ชัดกว่า S0 ที่ day 7 แต่ประโยชน์ลดลงมากที่ day 14 สำหรับ official alarm threshold 11.4 m จำนวนเหตุการณ์น้อยมากและ recall ที่ day 14 เป็นศูนย์ ผล event ทั้งหมดจึงจัดเป็น **exploratory evidence** เพราะแต่ละ fold มีเหตุการณ์อิสระน้อยกว่า 10 เหตุการณ์

## 7. ข้อสรุปทางวิจัย

1. กลยุทธ์ที่ freeze คือ MIMO และ configuration ที่ได้จาก validation คือ LSTM look-back 7 วัน
2. ข้อมูลหลายสถานีมีแนวโน้มช่วย horizon กลางถึงยาว แต่ไม่ได้ช่วย horizon สั้น
3. S4 ให้ improvement เฉลี่ย 5.35% ที่ day 7 แต่ paired CI ยังคร่อมศูนย์ จึงยังกล่าวไม่ได้ว่าดีกว่า S0 อย่างมีนัยสำคัญ
4. Ridge ยังคงเป็น baseline ที่แข่งขันได้สูง และผลไม่สนับสนุนคำกล่าวว่า deep learning เหนือกว่า classical model
5. predictively useful horizon ตามเกณฑ์ที่ preregister ไว้คือ 6 วัน
6. ประสิทธิภาพ event detection ที่ day 14 ยังไม่เพียงพอสำหรับข้อกล่าวอ้างเชิงปฏิบัติการ

## 8. Protocol deviation และข้อจำกัด

E2 ไม่ได้ใช้ Optuna 30 trials ต่อ architecture ตามร่าง methodology เนื่องจากไม่มี tuning implementation ใน starter suite การเลือก architecture จึงมาจาก frozen compact configurations และถูกบันทึกเป็น `COMPLETE_WITH_PROTOCOL_DEVIATION` ตั้งแต่ก่อนเปิด test

หลังเปิด test แล้วไม่ควรย้อนกลับไปทำ tuning และนำ configuration ใหม่มาประเมินบน test เดิม เพราะจะทำให้ test leakage เชิงการตัดสินใจ หากต้องการปิด deviation นี้อย่างเคร่งครัด ต้องกำหนด protocol ใหม่และใช้ข้อมูล prospective ที่ยังไม่ถูกเปิด เช่น complete 2026 holdout

### E2 Optuna 30 trials แบบ post-hoc

ตามคำขอภายหลัง ได้รัน Optuna เพิ่มเติมครบ 30 trials ต่อ architecture โดย objective ใช้เฉพาะค่า validation MAE เฉลี่ยที่ day 7 และ 14 จาก folds A–D ภายใต้ MIMO, S4, look-back 60 วัน และ development seed 42 ตัว objective ไม่ได้อ่าน test values แต่เนื่องจากผล E5 test ถูกเปิดไปแล้ว การทดลองนี้จึงเป็น **post-hoc validation-only sensitivity analysis** และไม่สามารถแทน architecture/configuration ที่ freeze ไว้เดิมได้

ช่วง search ใช้งบที่จำกัดตามทรัพยากร: LSTM และ TCN-GRU สูงสุด 25 epochs ต่อ fold (patience 5) ส่วน Compact Transformer สูงสุด 15 epochs (patience 4) จากนั้นนำ best configuration ของแต่ละ architecture มา refit ด้วยงบเท่ากันสูงสุด 100 epochs (patience 10)

| Architecture | Complete | Pruned | Search best MAE (m) | Full-refit MAE (m) |
|---|---:|---:|---:|---:|
| TCN-GRU | 18 | 12 | 0.777779 | **0.770217** |
| LSTM | 19 | 11 | 0.801961 | 0.787480 |
| Compact Transformer | 21 | 9 | 0.808075 | 0.804142 |

TCN-GRU ได้อันดับหนึ่งใน sensitivity analysis นี้ โดย best trial ใช้ `hidden_size=32`, `num_layers=2`, `conv_channels=64`, `kernel_size=3`, `dropout=0`, `batch_size=64` และ `learning_rate=0.0007541958` อย่างไรก็ตาม ห้ามนำผลนี้ไปย้อนแก้ E2 selection หรือประเมินซ้ำบน E5 test เดิมเพื่ออ้างผลยืนยัน หากต้องการยกระดับ TCN-GRU เป็นโมเดลหลัก ต้อง freeze configuration นี้ล่วงหน้าแล้วประเมินกับ prospective holdout ใหม่

### E3 ของ Optuna winner แบบ post-hoc

นำ TCN-GRU trial 15 จาก E2 Optuna มาประเมิน look-back 7, 14, 30 และ 60 วันบน validation folds A–D รวม 16 training jobs ทุก candidate ใช้ origins ที่ผ่านเงื่อนไขประวัติสูงสุด 60 วันเหมือนกัน และใช้ objective เป็น MAE เฉลี่ยที่ day 7 และ 14

| Look-back | Objective mean MAE (m) | Fold SE (m) | อยู่ใน one-SE ของค่าดีที่สุด |
|---:|---:|---:|---|
| 7 | 0.842822 | 0.015777 | ไม่ใช่ |
| 14 | 0.827126 | 0.017581 | ไม่ใช่ |
| 30 | 0.813334 | 0.025673 | ไม่ใช่ |
| 60 | **0.770217** | 0.023033 | ใช่ |

one-standard-error threshold เท่ากับ 0.793249 m จึงมีเพียง look-back 60 วันที่ผ่านเกณฑ์ และ E3 post-hoc เลือก **60 วัน** ผลนี้ต่างจาก frozen E3 ซึ่งเลือก LSTM look-back 7 วัน เพราะทั้ง architecture และ hyperparameters ต่างกัน จึงเป็น sensitivity result แยกชุด ไม่แทน E3/E4/E5 เดิมและไม่อนุญาตให้นำไปทดสอบซ้ำบน E5 test เดิมเพื่อกล่าวอ้างผลยืนยัน

การทดลอง direct-independent แบบ optional และ prospective 2026 holdout ไม่อยู่ในผลชุดนี้

## 9. การตรวจสอบและไฟล์ผลลัพธ์

- Regression tests ผ่าน 38/38
- E2 Optuna post-hoc trials: 90 (30 ต่อ architecture; failed/running 0)
- E3 Optuna-winner post-hoc training jobs: 16/16
- E2 training jobs: 12
- E3 training jobs: 16
- E4 training jobs: 36
- E5 neural training jobs: 50
- E5 Ridge jobs: 10
- E5 prediction rows ไม่มี duplicate และทุก target date อยู่หลัง issue date

ไฟล์หลัก:

- `artifacts/e2_architecture/E2_RESULTS.md`
- `artifacts/e2_optuna_posthoc/E2_OPTUNA_POSTHOC_RESULTS.md`
- `artifacts/e2_optuna_posthoc/architecture_ranking.csv`
- `artifacts/e2_optuna_posthoc/manifest.json`
- `artifacts/e2_optuna_posthoc/optuna_studies.sqlite3`
- `artifacts/e3_lookback/E3_RESULTS.md`
- `artifacts/e3_optuna_posthoc/E3_OPTUNA_POSTHOC_RESULTS.md`
- `artifacts/e3_optuna_posthoc/lookback_selection.csv`
- `artifacts/e3_optuna_posthoc/fold_objectives.csv`
- `artifacts/e3_optuna_posthoc/manifest.json`
- `artifacts/e4_stations/E4_RESULTS.md`
- `artifacts/e5_final/E5_RESULTS.md`
- `artifacts/e5_final/final_metric_summary.csv`
- `artifacts/e5_final/s4_vs_s0_inference.csv`
- `artifacts/e5_final/lstm_vs_ridge_inference.csv`
- `artifacts/e5_final/persistence_skill_inference.csv`
- `artifacts/e5_final/high_water_event_metrics.csv`
- `artifacts/e5_final/manifest.json`
