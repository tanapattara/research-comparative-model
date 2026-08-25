# รายงานสรุปและอธิบายผลการทดลอง E1

## การพยากรณ์ระดับน้ำแม่น้ำโขงที่สถานีหนองคายแบบหลายระยะนำ

**โครงการวิจัย:** Paper 2 — Mekong Multi-Horizon Forecasting  
**ระยะการทดลอง:** E1 — Paper 1 Continuity Baseline and Forecast Strategy Selection  
**สถานะ:** เสร็จสิ้นการประเมินบน validation เท่านั้น  
**วันที่จัดทำ:** 21 สิงหาคม 2026

> **ข้อควรระวังในการตีความ:** ผลในเอกสารนี้เป็นผลระยะพัฒนาจาก validation folds และ seed 42 เท่านั้น ยังไม่ใช่ผลวิจัยสุดท้าย ไม่มีการเปิดผล test partitions หรือ final-period test ปี 2025 และไม่มีการนำคะแนนจาก synthetic dataset มาตีความเป็นผลวิจัย

## 1. วัตถุประสงค์ของ E1

E1 มีวัตถุประสงค์เพื่อสร้างโมเดลที่ต่อเนื่องจาก Paper 1 และเลือกวิธีสร้างคำพยากรณ์ต่อเนื่อง 14 วันที่เหมาะสมสำหรับการทดลองขั้นถัดไป โดยเน้นการเปรียบเทียบระหว่างวิธี Direct multi-output หรือ MIMO กับวิธี Recursive ภายใต้เงื่อนไขที่ป้องกัน data leakage

E1 ยังไม่ได้มีวัตถุประสงค์เพื่อตัดสินว่าโมเดลใดดีที่สุดสำหรับงานวิจัยฉบับสุดท้าย และยังไม่ตอบคำถามเรื่องประโยชน์ของสถานีต้นน้ำแต่ละแห่ง เนื่องจากใช้เฉพาะ station set S4 เท่านั้น

## 2. สิ่งที่ดำเนินการใน E1

### 2.1 ข้อมูลและรูปแบบการพยากรณ์

ใช้ข้อมูลจริงรายวันจากสถานี CSA, LUA, CKH, VIE และ NON โดยกำหนด:

1. Input เป็นข้อมูลย้อนหลัง 60 วันจากทั้งห้าสถานี
2. Target เป็นระดับน้ำ NON ต่อเนื่องตั้งแต่ `t+1` ถึง `t+14`
3. รายงานผลหลักที่ระยะนำ 1, 3, 5, 7 และ 14 วัน
4. ใช้ forecast origins ชุดเดียวกันสำหรับทุกโมเดล
5. ตัด forecast origin ออก หาก input window หรือ NON ใน target window 14 วันมี missing

### 2.2 Temporal validation

E1 เปิดเฉพาะ validation partition ของ expanding-window folds ดังนี้:

| Fold | Training period | Validation period | Test period | Test opened in E1 |
|---|---|---|---|---|
| A | 2007–2016 | 2017 | 2018 | No |
| B | 2007–2018 | 2019 | 2020 | No |
| C | 2007–2020 | 2021 | 2022 | No |
| D | 2007–2022 | 2023 | 2024 | No |

มีการ purge forecast origins 14 วันก่อนสิ้นสุดแต่ละ partition เพื่อให้ target window ทั้ง 14 วันอยู่ภายใน partition เดียวกัน

### 2.3 การป้องกัน data leakage

มาตรการป้องกัน leakage ที่ใช้ประกอบด้วย:

1. Fit scaler จาก training partition เท่านั้น
2. ไม่ clip ค่า validation ที่อยู่นอกช่วงของ training scaler
3. ไม่เติม missing target ของ NON
4. ไม่ interpolate ข้อมูลทั้งชุดก่อนแบ่งเวลา
5. ใช้ common forecast origins จาก S4
6. Recursive LSTM พยากรณ์ค่าของทั้งห้าสถานีร่วมกัน และป้อนเฉพาะค่าพยากรณ์กลับเข้า input window
7. Recursive LSTM ไม่ใช้ค่าจริงของสถานีต้นน้ำหลัง issue time

### 2.4 โมเดลที่ประเมิน

โมเดลที่รันใน E1 ประกอบด้วย:

1. Persistence
2. Seasonal naive
3. Training-only day-of-year climatology
4. Direct multi-output Ridge
5. LSTM-MIMO
6. LSTM joint-recursive

### 2.5 โครงสร้าง LSTM ที่ต่อยอดจาก Paper 1

LSTM implementation ใหม่ใน Paper 2 ใช้ข้อกำหนดจาก manuscript ของ Paper 1 ได้แก่:

| Parameter | Value |
|---|---:|
| Input stations | CSA, LUA, CKH, VIE, NON |
| Look-back | 60 days |
| Number of LSTM layers | 2 |
| Hidden units per layer | 64 |
| Dropout | 0.2 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss | MSE |
| Batch size | 32 |
| Maximum epochs | 100 |
| Early-stopping patience | 10 |
| LR scheduler factor | 0.5 |
| LR scheduler patience | 5 |

MIMO ให้ผลลัพธ์ NON ทั้ง 14 วันพร้อมกัน ส่วน joint-recursive ให้ผลลัพธ์ของทั้งห้าสถานีทีละวันแล้วใช้ค่าพยากรณ์เหล่านั้นเป็น input สำหรับขั้นถัดไป

## 3. จำนวนตัวอย่างที่ใช้

หลังใช้ complete-case input window 60 วันและ target window 14 วัน จำนวน forecast origins แตกต่างกันระหว่าง folds ดังนี้:

| Fold | Training origins | Validation origins |
|---|---:|---:|
| A | 2,519 | 286 |
| B | 3,072 | 62 |
| C | 3,364 | 277 |
| D | 3,849 | 38 |

Fold B และ Fold D มี validation origins ค่อนข้างน้อย เนื่องจาก missing data ภายใน input window 60 วันหรือ target window 14 วัน ข้อจำกัดนี้ต้องนำมาพิจารณาเมื่อตีความค่าเฉลี่ยราย fold

## 4. ผล MAE ตามระยะนำ

ตารางต่อไปนี้รวม validation issue dates ทั้งสี่ folds โดยถ่วงตามจำนวน issue dates จริง หน่วยเป็นเมตร และค่าที่ต่ำกว่าหมายถึงผลดีกว่า

| Model | 1 day | 3 days | 5 days | 7 days | 14 days |
|---|---:|---:|---:|---:|---:|
| Persistence | 0.202 | 0.509 | 0.713 | 0.865 | 1.049 |
| Seasonal naive | 1.361 | 1.372 | 1.397 | 1.420 | 1.409 |
| Climatology | 1.493 | 1.492 | 1.500 | 1.510 | 1.526 |
| Ridge | **0.124** | 0.316 | 0.480 | 0.652 | **0.901** |
| LSTM-MIMO | 0.352 | 0.365 | 0.484 | 0.670 | 0.968 |
| LSTM joint-recursive | 0.129 | **0.288** | **0.454** | **0.648** | 1.009 |

### 4.1 ข้อสังเกตจาก MAE

1. **ระยะนำ 1 วัน:** Ridge ให้ MAE ต่ำที่สุดที่ 0.124 เมตร ขณะที่ LSTM-MIMO แย่กว่า persistence
2. **ระยะนำ 3–7 วัน:** Joint-recursive LSTM ให้ MAE ต่ำที่สุดเล็กน้อย
3. **ระยะนำ 14 วัน:** Ridge ให้ MAE ต่ำที่สุดที่ 0.901 เมตร ตามด้วย MIMO ที่ 0.968 เมตร
4. Seasonal naive และ climatology แย่กว่า persistence อย่างชัดเจนในข้อมูลชุดนี้
5. MAE เพิ่มขึ้นตามระยะนำในทุกโมเดล สะท้อนว่าความไม่แน่นอนเพิ่มขึ้นเมื่อพยากรณ์ไกลขึ้น

ผลดังกล่าวแสดงว่า Ridge เป็น baseline ที่แข็งแรงมาก และยังไม่มีหลักฐานจาก E1 ว่า deep learning ดีกว่า Ridge

## 5. MAE Skill เทียบ Persistence

MAE skill ที่เป็นค่าบวกหมายถึงโมเดลลด MAE จาก persistence ได้ เช่น skill เท่ากับ 0.25 หมายถึงลด MAE ได้ประมาณร้อยละ 25

| Horizon | Ridge | LSTM-MIMO | LSTM joint-recursive |
|---|---:|---:|---:|
| 1 day | +38.6% | −87.6% | +35.7% |
| 3 days | +37.9% | +26.7% | +43.4% |
| 5 days | +32.7% | +32.0% | +35.9% |
| 7 days | +25.0% | +23.1% | +24.7% |
| 14 days | +13.8% | +6.9% | +4.6% |

ข้อได้เปรียบเหนือ persistence ลดลงเมื่อระยะนำเพิ่มขึ้น โดยที่ 14 วัน Ridge ยังลด MAE ได้ประมาณร้อยละ 13.8 ส่วน MIMO ลดได้ประมาณร้อยละ 6.9

ผลนี้เป็นหลักฐานเบื้องต้นว่าการพยากรณ์ที่ 14 วันอาจยังให้ข้อมูลเพิ่มจาก persistence แต่ยังไม่เพียงพอสำหรับสรุปว่าระยะนำ 14 วันมีความน่าเชื่อถือในเชิงปฏิบัติ จนกว่าจะผ่านการประเมินบน test folds, หลาย seeds และการวิเคราะห์ uncertainty ที่ครบถ้วน

## 6. การเปรียบเทียบ MIMO กับ Recursive

การเลือก forecast strategy ใช้ paired absolute errors จาก issue dates เดียวกันจำนวน 663 วันต่อ horizon และพิจารณาระยะนำ 7 กับ 14 วันเป็นหลัก

### 6.1 ผลรายระยะนำ

| Horizon | Mean MAE difference: MIMO − Recursive | Interpretation |
|---|---:|---|
| 7 days | +0.0221 m | Recursive ต่ำกว่า MIMO เล็กน้อย |
| 14 days | −0.0410 m | MIMO ต่ำกว่า Recursive |

เมื่อรวมระยะนำ 7 และ 14 วัน:

- LSTM-MIMO MAE = 0.819217 เมตร
- LSTM joint-recursive MAE = 0.828660 เมตร
- Recursive แย่กว่า MIMO ประมาณร้อยละ 1.15

### 6.2 Moving-block bootstrap และ Diebold–Mariano test

ใช้ moving-block bootstrap 2,000 ครั้ง โดยมี block length หลัก 30 วัน และ sensitivity ที่ 14 กับ 60 วัน

| Analysis | 95% confidence interval |
|---|---:|
| Combined horizons 7/14, block 14 days | [−0.0764, 0.0490] m |
| Combined horizons 7/14, block 30 days | [−0.0918, 0.0349] m |
| Combined horizons 7/14, block 60 days | [−0.0889, 0.0201] m |

ช่วงความเชื่อมั่นทุกกรณีคร่อมศูนย์ แสดงว่ายังไม่มีหลักฐานเพียงพอว่า recursive ดีกว่า MIMO

Diebold–Mariano tests ใช้ HAC lag เท่ากับ `h−1` และ Holm correction สำหรับผลวันที่ 7 และ 14 ได้ adjusted p-value เท่ากับ 0.8199 จึงไม่พบความแตกต่างที่มีนัยสำคัญ

นอกจากนี้ recursive ไม่ผ่านเกณฑ์เชิงปฏิบัติที่กำหนดว่าต้องลด MAE อย่างน้อยร้อยละ 5

## 7. เหตุผลที่เลือก MIMO เป็นวิธีหลัก

แม้ recursive จะให้ MAE ต่ำกว่าเล็กน้อยในระยะ 3–7 วัน แต่ผลดังกล่าวไม่สม่ำเสมอ และ recursive แย่ลงที่ระยะ 14 วัน จึงเลือก MIMO เป็น strategy หลักสำหรับการทดลองขั้นถัดไปด้วยเหตุผลดังนี้:

1. เป็น forecast strategy หลักที่กำหนดไว้ล่วงหน้าใน protocol
2. สร้างผลพยากรณ์ทั้ง 14 วันพร้อมกันจากข้อมูลที่มีอยู่ ณ issue time
3. ไม่มี error feedback ทีละวันเหมือน recursive
4. ไม่ต้องอาศัยค่าจริงของสถานีต้นน้ำหลัง issue time
5. ใช้งานจริงและตรวจสอบ leakage ได้ง่ายกว่า
6. Recursive ไม่ผ่านทั้งเกณฑ์ confidence interval และเกณฑ์ลด MAE อย่างน้อยร้อยละ 5

การเลือก MIMO ใน E1 หมายถึงการเลือก **forecast strategy** ไม่ได้หมายความว่า LSTM-MIMO เป็นโมเดลที่ดีที่สุด เนื่องจาก Ridge ยังมีผลดีกว่า MIMO หลาย horizon

## 8. ความแตกต่างระหว่างค่าเฉลี่ยแบบ Fold กับแบบ Issue Date

ค่าเฉลี่ยระดับ fold แบบให้น้ำหนักแต่ละ fold เท่ากันเคยทำให้ recursive ดูดีกว่า MIMO ประมาณร้อยละ 2.57 อย่างไรก็ตาม folds มีจำนวน validation origins ไม่เท่ากันอย่างมาก โดยเฉพาะ Fold B มี 62 origins และ Fold D มี 38 origins

การตัดสินใจขั้นสุดท้ายของ E1 จึงใช้ paired errors ระดับ issue date ตามข้อกำหนดของ protocol ส่งผลให้ MIMO มี MAE 0.819217 เมตร และ recursive มี MAE 0.828660 เมตร ความแตกต่างนี้เกิดจากวิธีถ่วงน้ำหนัก ไม่ใช่จากการเปลี่ยน prediction หรือเลือกเฉพาะผลที่ต้องการ

## 9. ข้อจำกัดของ E1

1. ใช้ station set S4 เท่านั้น จึงยังไม่ตอบว่าสถานีใดมีประโยชน์ในแต่ละ horizon
2. ใช้ look-back 60 วันเท่านั้น ยังไม่ได้ทดลอง 7, 14 และ 30 วัน
3. ใช้ development seed 42 เพียง seed เดียว
4. จำนวน validation origins แตกต่างกันมากระหว่าง folds
5. ยังไม่ได้ประเมิน TCN-GRU และ Compact Transformer
6. ยังไม่ได้เปิด test partitions หรือ final-period test ปี 2025
7. ยังไม่ได้วิเคราะห์ผลเฉพาะช่วงน้ำสูง เหตุการณ์วิกฤต หรือ threshold exceedance
8. การทดสอบทางสถิติใน E1 เปรียบเทียบ MIMO กับ recursive เท่านั้น ยังไม่ได้ทำ paired inference ระหว่าง Ridge, persistence และ neural models อย่างครบถ้วน

## 10. ข้อสรุปและข้อเสนอแนะสำหรับขั้นถัดไป

E1 แสดงว่า pipeline สามารถฝึกและประเมินโมเดลแบบ multi-horizon โดยรักษา temporal integrity และป้องกัน future-input leakage ได้สำเร็จ

ข้อสรุปที่นำไปใช้ต่อคือ:

1. Freeze MIMO เป็น forecast strategy หลักสำหรับการทดลองขั้นถัดไป
2. เก็บ joint-recursive เป็น comparator เท่านั้น
3. ใช้ Ridge เป็น baseline สำคัญ เพราะให้ผลแข่งขันได้ดีมาก
4. ยังไม่ควรกล่าวว่า deep learning เหนือกว่า classical model
5. ต้องทดลอง look-back 7, 14, 30 และ 60 วันโดยใช้ validation เท่านั้นก่อน freeze hyperparameters
6. ต้องทำ station incremental และ leave-one-out experiments โดย retrain weights แต่ไม่ retune hyperparameters
7. Final neural runs ต้องใช้ seeds 42, 52, 62, 72 และ 82 และต้องรายงานครบทุก seed
8. ต้องเก็บ test partitions ปิดไว้จนกว่าจะ freeze strategy และ hyperparameters

## 11. ไฟล์หลักที่ใช้จัดทำรายงาน

- `artifacts/e1_strategy/validation_metrics.csv`
- `artifacts/e1_strategy/validation_predictions.csv`
- `artifacts/e1_strategy/training_summary.csv`
- `artifacts/e1_strategy/strategy_inference.csv`
- `artifacts/e1_strategy/strategy_decision.json`
- `reports/e1_strategy_summary.md`
- `reports/paper1_code_alignment.md`

---

**สถานะหลัง E1:** เลือก LSTM-MIMO เป็น forecast strategy หลัก แต่ยังไม่เริ่ม E2, station ablation, TCN-GRU, Compact Transformer หรือ final neural evaluation
