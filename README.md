#เครื่องนับเม็ดยาอัจฉริยะระบบขอบข่ายด้วยการเรียนรู้ของเครื่อง
### (Edge-Based Machine Learning Smart Pill Counter)

[![NSC 28th](https://img.shields.io/badge/NSC-28th_Contest-blue.svg)](https://www.nectec.or.th/)
[![Project ID](https://img.shields.io/badge/Project_ID-28P14E01386-orange.svg)]()
[![License](https://img.shields.io/badge/License-Confidential-red.svg)]()
[![Platform](https://img.shields.io/badge/Platform-Raspberry_Pi_5-green.svg)]()
[![Model](https://img.shields.io/badge/Model-YOLOv8n-brightgreen.svg)]()

> **การแข่งขันพัฒนาโปรแกรมคอมพิวเตอร์แห่งประเทศไทย ครั้งที่ 28 (NSC 28th)**  
> **ระดับ:** นิสิต นักศึกษา | **หมวด:** โปรแกรมเพื่องานการพัฒนาด้านวิทยาศาสตร์และเทคโนโลยี  
> **สังกัด:** ภาควิชาวิศวกรรมคอมพิวเตอร์และสารสนเทศศาสตร์ คณะวิศวกรรมศาสตร์ศรีราชา มหาวิทยาลัยเกษตรศาสตร์ วิทยาเขตศรีราชา  

---

## 📌 บทนำและที่มาของโครงการ (Project Overview)

ปัญหา **ความคลาดเคลื่อนทางยา (Medication Error)** ในขั้นตอนการจัดและจ่ายยา (Pre-dispensing & Dispensing Error) เป็นหนึ่งในสาเหตุสำคัญที่ส่งผลกระทบโดยตรงต่อความปลอดภัยของผู้ป่วย (Patient Safety) ซึ่งมักเกิดจากภาระงานที่ล้นมือและความเหนื่อยล้าสะสมของบุคลากรทางการแพทย์ (Human Error)

โครงการ **"เครื่องนับเม็ดยาอัจฉริยะระบบขอบข่ายด้วยการเรียนรู้ของเครื่อง (Edge-Based Machine Learning Smart Pill Counter)"** พัฒนาขึ้นเพื่อทำหน้าที่เป็น **ระบบตรวจสอบซ้ำอัตโนมัติ (Double Check Artifact)** ที่มีความแม่นยำสูง ประมวลผลแบบ Standalone / Offline บนอุปกรณ์ระดับขอบข่าย (Edge Computing) โดยไม่ต้องพึ่งพาระบบอินเทอร์เน็ต เพื่อความรวดเร็วในการประมวลผลและความเป็นส่วนตัวของข้อมูล

---

## 🚀 คุณสมบัติเด่น (Key Features)

- **On-Device Edge AI Processing:** ประมวลผลสตรีมภาพและนับจำนวนเม็ดยาแบบ Real-time ณ จุดใช้งาน (Low Latency)
- **Multi-Class Pill Detection:** ตรวจจับและจำแนกรูปทรงเม็ดยามาตรฐานได้ 3 ประเภทหลัก:
  - 🔵 **ทรงกลม (Round Tablet)**
  - 🟢 **ทรงรี (Oval Tablet)**
  - 💊 **แคปซูล (Capsule)**
- **Overlapping Elimination Mechanism (กลไกแก้ปัญหายาซ้อนทับ):**
  - วิเคราะห์พื้นที่ทับซ้อนด้วยสมการ **Intersection over Union (IoU)**
  - สั่งการ **มอเตอร์สั่นถาดนับยาอัตโนมัติ (Tray Vibration Motor)** เมื่อพบเม็ดยาวางทับซ้อนกัน เพื่อสลายกลุ่มก้อนยา[cite: 2]
- **Standalone & Offline Operation:** ทำงานได้โดยไม่ต้องเชื่อมต่ออินเทอร์เน็ต แสดงผลผ่านหน้าจอ LCD / Web Application
- **Patient Safety Support:** ส่งสัญญาณเสียงเตือน (Buzzer) เมื่อกระบวนการนับเสร็จสิ้น หรือตรวจพบความคลาดเคลื่อน[cite: 2]

---

## 🔄 การพัฒนาและปรับเปลี่ยนสถาปัตยกรรม (Hardware & Model Migration)

ในการพัฒนาระยะแรก ได้มีการออกแบบระบบบนบอร์ด **Sipeed Maix Go (ชิปเซ็ต Kendryte K210)** ร่วมกับ **MicroPython / Tiny-YOLO (YOLOv2)**[cite: 2] อย่างไรก็ตาม จากการทดสอบใช้งานจริงพบข้อจำกัดทางเทคนิค:

- ⚠️ **Hardware Bottleneck:** SRAM บนชิปเซ็ต K210 มีจำกัด (ใช้งานได้จริงประมาณ 6MB) ทำให้แรมไม่เพียงพอสำหรับการประมวลผลโมเดล Object Detection ที่มีความแม่นยำสูงและการจัดการเฟรมภาพหลายคลาส[cite: 2]
- ✅ **Architectural Upgrade:** ทางทีมพัฒนาจึงได้ทำการปรับเปลี่ยนสถาปัตยกรรมฮาร์ดแวร์มาใช้ **Raspberry Pi 5** ร่วมกับ **Webcam** และอัปเกรดโมเดลเป็น **YOLOv8n**
- 📈 **Performance Gain:** การอัปเกรดเป็น Raspberry Pi 5 + YOLOv8n ช่วยเพิ่มความแม่นยำในการตรวจจับ (Precision/Recall), รองรับการประมวลผล Real-time Frame Rate ที่สูงขึ้น และทำให้ระบบตรวจจับความทับซ้อน (IoU) สั่งงานมอเตอร์สั่นได้อย่างเสถียร

---

## 🏗️ สถาปัตยกรรมระบบ (System Architecture)

```text
                     ┌─────────────────────────┐
                     │   High-Res Webcam /     │
                     │     Camera Module       │
                     └────────────┬────────────┘
                                  │ (Capture Live Frame)
                                  ▼
               ┌──────────────────────────────────────┐
               │         Raspberry Pi 5               │
               │  ┌────────────────────────────────┐  │
               │  │ Image Preprocessing & Resize   │  │
               │  └───────────────┬────────────────┘  │
               │                  ▼                   │
               │  ┌────────────────────────────────┐  │
               │  │ YOLOv8n Inference Engine       │  │
               │  │ (Round, Oval, Capsule Detection)│  │
               │  └───────────────┬────────────────┘  │
               │                  ▼                   │
               │  ┌────────────────────────────────┐  │
               │  │ Overlap Detection (IoU Check)  │  │
               │  │ & Counting Logic               │  │
               │  └───────┬──────────────┬─────────┘  │
               └──────────┼──────────────┼────────────┘
                          │              │
         ┌────────────────┘              └────────────────┐
         ▼                                                ▼
┌─────────────────────────┐                      ┌─────────────────────────┐
│ LCD Display / Web UI    │                      │ Hardware Control        │
│ - Live Feed & Boxes     │                      │ - PWM Vibration Motor   │
│ - Total Pill Count      │                      │ - Alert Buzzer          │
└─────────────────────────┘                      └─────────────────────────┘
